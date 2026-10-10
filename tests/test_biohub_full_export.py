"""Checks for streamed center extraction and a round-trip GEFF graph."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from importlib.util import find_spec

import numpy as np
import zarr

from linajea.biohub_full_export import (_frame_candidates, solve_components,
                                        write_predicted_geff)
from linajea.biohub_io import read_geff_tracks
from linajea.biohub_reconstruct import candidate_edges


def test_frame_candidates_keep_global_voxel_positions():
    ci = np.zeros((4, 12, 12), dtype=np.float32)
    ci[2, 4, 5] = 0.9
    mv = np.zeros((3,) + ci.shape, dtype=np.float32)
    mv[2, 2, 4, 5] = -32
    rows = _frame_candidates(ci, mv, 3, (3, 3, 3), 0.2, 200, 10)
    assert len(rows) == 1
    assert (rows[0]["node_id"], rows[0]["t"], rows[0]["z"],
            rows[0]["y"], rows[0]["x"]) == (10, 3, 2, 4, 5)
    assert rows[0]["mv_x_um"] == -1


def test_component_ilp_retains_two_independent_tracks():
    nodes = [{"node_id": i, "t": i % 2, "z": 2,
              "y": 2 + 40 * (i // 2), "x": 2,
              "score": 0.9, "mv_z_um": 0,
              "mv_y_um": 0, "mv_x_um": 0} for i in range(4)]
    edges = candidate_edges(nodes)
    chosen_nodes, chosen_edges, statuses = solve_components(
        nodes, edges, selection_constant=4, track_cost=1)
    assert len(statuses) == 2
    assert len(chosen_nodes) == 4
    assert len(chosen_edges) == 2


def test_predicted_geff_round_trip():
    nodes = [{"node_id": i, "t": i, "z": 2, "y": 3, "x": 4}
             for i in range(2)]
    edges = [{"source_id": 0, "target_id": 1}]
    metadata = {"geff_version": "1.1", "directed": True,
                "axes": [{"name": name, "type": kind,
                          "scale": scale, "min": 0, "max": 9,
                          "unit": unit}
                         for name, kind, scale, unit in (
                             ("t", "time", 1, "frame"),
                             ("z", "space", 1.625, "micrometer"),
                             ("y", "space", .40625, "micrometer"),
                             ("x", "space", .40625, "micrometer"))],
                "node_props_metadata": {axis: {"dtype": "int64",
                                               "identifier": axis}
                                        for axis in ("t", "z", "y", "x")},
                "edge_props_metadata": {},
                "extra": {}}

    class FakeImage:
        shape = (3, 8, 8, 8)
        voxel_size = (1, 52, 13, 13)

    with TemporaryDirectory() as directory:
        root = Path(directory)
        original = zarr.open_group(str(root / "original.geff"), mode="w",
                                   zarr_format=2)
        original.attrs["geff"] = metadata
        path = root / "predicted.geff"
        with patch("linajea.biohub_full_export.BiohubImage",
                   return_value=FakeImage()):
            write_predicted_geff(path, nodes, edges,
                                 root / "original.geff", root / "image.zarr")
        records, edge_count = read_geff_tracks(path, FakeImage.voxel_size,
                                               FakeImage.shape)
        assert len(records) == 2 and edge_count == 1
        assert records[1]["parent_id"] == 0
        assert (path / ".zgroup").is_file()
        if find_spec("geff"):
            import geff
            geff.validate_structure(str(path))
            graph, _ = geff.read(str(path))
            assert graph.number_of_nodes() == 2
            assert graph.number_of_edges() == 1

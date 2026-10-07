"""Toy graph checks for held-out matching and the Linajea basic ILP."""

import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from linajea.biohub_calibrate import calibrate_cached_candidates
from linajea.biohub_random_hide import match_nodes
from linajea.biohub_random_hide import write_rows
from linajea.biohub_reconstruct import candidate_edges, solve_basic_ilp
from linajea.biohub_reconstruct import EDGE_FIELDS, NODE_FIELDS


def test_physical_matching_is_one_to_one():
    gt = [
        {"cell_id": 10, "t": 0, "z": 0, "y": 0, "x": 0},
        {"cell_id": 11, "t": 0, "z": 0, "y": 0, "x": 13},
    ]
    predicted = [
        {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0},
        {"node_id": 2, "t": 0, "z": 0, "y": 0, "x": 0},
    ]
    matches = match_nodes(predicted, gt, 0.2)
    assert len(matches) == 1
    assert set(matches.values()) == {10}


def test_matching_maximizes_recovered_nodes_before_distance():
    gt = [
        {"cell_id": 10, "t": 0, "z": 0, "y": 0, "x": 0},
        {"cell_id": 11, "t": 0, "z": 0, "y": 0, "x": 64},
    ]
    predicted = [
        {"node_id": 1, "t": 0, "z": 0, "y": 0, "x": 0},
        {"node_id": 2, "t": 0, "z": 0, "y": 0, "x": -4},
    ]
    assert match_nodes(predicted, gt, 2.0) == {1: 11, 2: 10}


def test_candidate_edges_and_basic_ilp_select_chain():
    nodes = [{"node_id": t, "t": t, "z": 10, "y": 20, "x": 30,
              "score": 0.95, "mv_z_um": 0, "mv_y_um": 0,
              "mv_x_um": 0} for t in range(5)]
    edges = candidate_edges(nodes)
    assert len(edges) == 4
    assert [(e["source_id"], e["target_id"]) for e in edges] == [
        (0, 1), (1, 2), (2, 3), (3, 4)]
    selected_nodes, selected_edges, status = solve_basic_ilp(nodes, edges)
    assert len(selected_nodes) == 5
    assert len(selected_edges) == 4
    assert status["selected_edges"] == 4


def test_visible_only_calibration_resolves_empty_graph():
    nodes = [{"node_id": t, "t": t, "z": 10, "y": 20, "x": 30,
              "score": 0.5, "mv_z_um": 0, "mv_y_um": 0,
              "mv_x_um": 0} for t in range(5)]
    nodes.append({"node_id": 5, "t": 0, "z": 10, "y": 100, "x": 100,
                  "score": 0.2, "mv_z_um": 0, "mv_y_um": 0,
                  "mv_x_um": 0})
    edges = candidate_edges(nodes)
    assert not solve_basic_ilp(nodes, edges)[0]
    visible = [{"cell_id": t, "parent_id": t - 1 if t else -1,
                "t": t, "z": 520, "y": 260, "x": 390}
               for t in range(5)]
    with TemporaryDirectory() as directory:
        root = Path(directory)
        candidate_dir = root / "candidate"
        candidate_dir.mkdir()
        write_rows(candidate_dir / "candidate_nodes.csv", NODE_FIELDS, nodes)
        write_rows(candidate_dir / "candidate_edges.csv", EDGE_FIELDS, edges)
        with patch("linajea.biohub_calibrate._visible_roi_records",
                   return_value=visible):
            report = calibrate_cached_candidates(
                root / "unused.zarr", root / "visible.geff",
                root / "split.json", candidate_dir, root / "calibrated")
        assert report["hidden_gt_used_for_selection"] is False
        assert report["chosen"]["visible_node_recovered"] == 5
        assert report["chosen"]["visible_edge_recovered"] == 4
        assert report["chosen"]["selected_nodes"] == 5
        assert json.loads((root / "calibrated" / "calibration.json").read_text())[
            "chosen"]["selected_edges"] == 4

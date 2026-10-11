"""Toy checks for fixed-cost and two-anchor pseudo-label diagnostics."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json

import numpy as np
import zarr

from linajea.biohub_edge_audit import audit_temporal_edges
from linajea.biohub_pseudo_analysis import (anchored_corridor,
                                            analyze_cached,
                                            candidate_frame_diagnostics,
                                            discover_runs)
from linajea.biohub_random_hide import write_rows
from linajea.biohub_reconstruct import NODE_FIELDS, EDGE_FIELDS


def _chain():
    nodes = [{"node_id": t, "t": t, "z": 2, "y": 5, "x": 7,
              "score": 0.9} for t in range(5)]
    edges = [{"source_id": t, "target_id": t + 1,
              "prediction_distance_um": 1.0} for t in range(4)]
    visible = [{"cell_id": 10, "t": 0, "z": 104,
                "y": 65, "x": 91},
               {"cell_id": 11, "t": 4, "z": 104,
                "y": 65, "x": 91}]
    return nodes, edges, visible


def test_two_anchor_path_keeps_only_interior_pseudo_nodes():
    nodes, edges, visible = _chain()
    graph_nodes, graph_edges, pseudo, info = anchored_corridor(
        nodes, edges, visible, max_missing_frames=3)
    assert [int(node["node_id"]) for node in pseudo] == [1, 2, 3]
    assert len(graph_nodes) == 5 and len(graph_edges) == 4
    assert info["matched_visible_anchors"] == 2


def test_anchor_path_requires_score_and_short_gap():
    nodes, edges, visible = _chain()
    nodes[2]["score"] = 0.25
    _, links, pseudo, _ = anchored_corridor(
        nodes, edges, visible, min_score=0.4)
    assert not pseudo and not links
    nodes[2]["score"] = 0.9
    _, links, pseudo, _ = anchored_corridor(
        nodes, edges, visible, max_missing_frames=2)
    assert not pseudo and not links


def test_candidate_cap_is_reported_per_frame():
    nodes = [{"t": 7, "score": 0.3} for _ in range(200)]
    rows = candidate_frame_diagnostics(nodes, 7, 2)
    assert rows[0]["candidate_count"] == 200
    assert rows[0]["at_candidate_cap"] is True
    assert rows[1]["candidate_count"] == 0


def test_temporal_edge_audit_separates_missing_center_from_missing_edge():
    gt = [{"cell_id": 10 + t, "parent_id": 9 + t if t else -1,
           "t": t, "z": 52, "y": 130, "x": 130} for t in range(3)]
    nodes = [{"node_id": 20 + t, "t": t, "z": 1, "y": 10, "x": 10,
              "score": 0.8} for t in range(3)]
    edges = [{"source_id": 20, "target_id": 21,
              "prediction_distance_um": 0.5}]
    manifest = {"hidden_node_ids": [11],
                "hidden_edges": [[10, 11], [11, 12]]}
    predicted, known, localization = audit_temporal_edges(
        nodes, edges, gt, [gt[0], gt[2]], manifest,
        "seed_42", "anchor_strict")
    assert predicted[0]["gt_relation"] == "confirmed_hidden_edge"
    assert predicted[0]["source_error_um"] == 0
    assert [row["failure_reason"] for row in known] == [
        "recovered", "edge_missing_despite_both_centers"]
    assert localization[1]["hidden_matched"] == 1
    _, known_missing, _ = audit_temporal_edges(
        [nodes[0], nodes[2]], [], gt, [gt[0], gt[2]], manifest,
        "seed_42", "anchor_strict")
    assert [row["failure_reason"] for row in known_missing] == [
        "target_center_missing", "source_center_missing"]


def test_discovers_multiseed_cached_output():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "seed_42"
        (run / "prediction").mkdir(parents=True)
        (run / "split.json").touch()
        (run / "visible.geff").mkdir()
        (run / "prediction" / "candidate_nodes.csv").touch()
        (run / "prediction" / "candidate_edges.csv").touch()
        assert list(discover_runs(root)) == ["seed_42"]


def test_discovers_checkpoint_1500_cached_output():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        candidate = root / "checkpoint_comparison" / "iteration_1500"
        candidate.mkdir(parents=True)
        (root / "split.json").touch()
        (root / "visible.geff").mkdir()
        (candidate / "candidate_nodes.csv").touch()
        (candidate / "candidate_edges.csv").touch()
        assert list(discover_runs(root)) == ["checkpoint_1500"]


def test_cached_analysis_keeps_hidden_gt_out_of_selection():
    nodes, edges, _ = _chain()
    for node in nodes:
        node.update({"mv_z_um": 0, "mv_y_um": 0, "mv_x_um": 0})
    with TemporaryDirectory() as directory:
        base = Path(directory)
        source = base / "previous"
        run = source / "seed_42"
        (run / "prediction").mkdir(parents=True)
        (run / "split.json").write_text(json.dumps({
            "sample": "toy", "roi_start": [0, 0, 0, 0],
            "roi_shape": [5, 8, 8, 8], "hidden_node_ids": [],
            "hidden_edges": []}))
        write_rows(run / "prediction" / "candidate_nodes.csv",
                   NODE_FIELDS, nodes)
        write_rows(run / "prediction" / "candidate_edges.csv",
                   EDGE_FIELDS, edges)
        geff = zarr.open_group(str(run / "visible.geff"), mode="w",
                               zarr_format=2)
        geff.attrs["geff"] = {
            "geff_version": "1.1", "directed": True,
            "axes": [{"name": axis, "type": kind, "scale": scale,
                      "unit": unit, "min": 0, "max": 8}
                     for axis, kind, scale, unit in (
                         ("t", "time", 1, "frame"),
                         ("z", "space", 1.625, "micrometer"),
                         ("y", "space", 0.40625, "micrometer"),
                         ("x", "space", 0.40625, "micrometer"))],
            "node_props_metadata": {axis: {"dtype": "int64",
                                           "identifier": axis}
                                    for axis in ("t", "z", "y", "x")},
            "edge_props_metadata": {}, "extra": {}}
        geff.create_array("nodes/ids", data=np.array([10, 11], dtype="uint64"))
        geff.create_array("edges/ids", data=np.empty((0, 2), dtype="uint64"))
        for axis, values in {"t": [0, 4], "z": [2, 2],
                             "y": [5, 5], "x": [7, 7]}.items():
            geff.create_array(f"nodes/props/{axis}/values",
                              data=np.array(values, dtype="int64"))
        summary = analyze_cached(source, base / "analysis")
        by_stage = {row["stage"]: row for row in summary}
        assert by_stage["candidate"]["node_count"] == 5
        assert by_stage["fixed_ilp"]["node_count"] == 5
        assert by_stage["anchor_strict"]["pseudo_node_count"] == 3
        assert by_stage["anchor_strict"]["hidden_recovered"] is None
        assert (base / "analysis" / "audit_queue.csv").is_file()

        gt_dir = base / "gt"
        gt_dir.mkdir()
        original = zarr.open_group(str(gt_dir / "toy.geff"), mode="w",
                                   zarr_format=2)
        original.attrs["geff"] = dict(geff.attrs["geff"])
        original_ids = np.array([10, 12, 13, 14, 11], dtype="uint64")
        original.create_array("nodes/ids", data=original_ids)
        original.create_array("edges/ids", data=np.array(
            [[10, 12], [12, 13], [13, 14], [14, 11]], dtype="uint64"))
        for axis, values in {"t": [0, 1, 2, 3, 4], "z": [2] * 5,
                             "y": [5] * 5, "x": [7] * 5}.items():
            original.create_array(f"nodes/props/{axis}/values",
                                  data=np.array(values, dtype="int64"))
        (run / "split.json").write_text(json.dumps({
            "sample": "toy", "roi_start": [0, 0, 0, 0],
            "roi_shape": [5, 8, 8, 8], "hidden_node_ids": [12, 13, 14],
            "hidden_edges": [[10, 12], [12, 13], [13, 14], [14, 11]]}))
        with_gt = analyze_cached(source, base / "analysis_with_gt", gt_dir)
        by_stage = {row["stage"]: row for row in with_gt}
        assert by_stage["anchor_strict"]["hidden_recovered"] == 3
        assert by_stage["anchor_strict"]["hidden_edge_recovered"] == 4
        assert by_stage["anchor_strict"]["unconfirmed_after_full_gt"] == 0
        assert (base / "analysis_with_gt" / "predicted_edge_audit.csv").is_file()
        assert (base / "analysis_with_gt" / "gt_edge_audit.csv").is_file()
        assert (base / "analysis_with_gt" / "localization_audit.csv").is_file()

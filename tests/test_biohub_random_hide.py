"""Toy graph checks for held-out matching and the Linajea basic ILP."""

import json
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from linajea.biohub_calibrate import calibrate_cached_candidates
from linajea.biohub_checkpoint_compare import _metrics, compare_checkpoints
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


def test_checkpoint_metrics_separate_visible_and_hidden():
    visible = {"cell_id": 10, "parent_id": -1, "t": 0,
               "z": 0, "y": 0, "x": 0}
    hidden = {"cell_id": 11, "parent_id": 10, "t": 1,
              "z": 0, "y": 0, "x": 0}
    nodes = [{"node_id": 0, "t": 0, "z": 0, "y": 0, "x": 0},
             {"node_id": 1, "t": 1, "z": 0, "y": 0, "x": 0}]
    edges = [{"source_id": 0, "target_id": 1}]
    result = _metrics(500, "candidate", nodes, edges,
                      [visible, hidden], [visible], {11}, set(),
                      {(10, 11)}, 4.0)
    assert result["visible_recovered"] == 1
    assert result["hidden_recovered"] == 1
    assert result["hidden_edges_recovered"] == 1


def test_checkpoint_comparison_reuses_final_prediction():
    nodes = [{"node_id": t, "t": t, "z": 0, "y": 0, "x": 0,
              "score": 0.9, "mv_z_um": 0, "mv_y_um": 0,
              "mv_x_um": 0} for t in range(2)]
    edges = candidate_edges(nodes)
    visible = {"cell_id": 10, "parent_id": -1, "t": 0,
               "z": 0, "y": 0, "x": 0}
    hidden = {"cell_id": 11, "parent_id": 10, "t": 1,
              "z": 0, "y": 0, "x": 0}

    class FakeImage:
        voxel_size = (1, 52, 13, 13)
        shape = (2, 1, 1, 1)

    def fake_tracks(path, *_):
        return ([visible] if Path(path).name == "visible.geff"
                else [visible, hidden]), 1

    def fake_predict(_, __, ___, ____, output_dir, **kwargs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_rows(output_dir / "candidate_nodes.csv", NODE_FIELDS, nodes)
        return nodes, {"candidate_count": 2}

    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "train").mkdir()
        for iteration in (1, 2):
            (root / "train" / f"train_net_checkpoint_{iteration}").touch()
        (root / "train" / "config.toml").touch()
        (root / "split.json").write_text(json.dumps({
            "sample": "toy", "roi_start": [0, 0, 0, 0],
            "roi_shape": [2, 1, 1, 1], "hidden_node_ids": [11],
            "hidden_edges": [[10, 11]]}))
        (root / "prediction").mkdir()
        (root / "prediction_calibrated").mkdir()
        write_rows(root / "prediction" / "candidate_nodes.csv",
                   NODE_FIELDS, nodes)
        write_rows(root / "prediction" / "candidate_edges.csv",
                   EDGE_FIELDS, edges)
        write_rows(root / "prediction_calibrated" / "pred_nodes.csv",
                   NODE_FIELDS, nodes)
        write_rows(root / "prediction_calibrated" / "pred_edges.csv",
                   EDGE_FIELDS, edges)
        (root / "prediction_calibrated" / "calibration.json").write_text(
            json.dumps({"chosen": {"selection_constant": 5.0,
                                   "track_cost": 1.0}}))
        with patch("linajea.biohub_checkpoint_compare.BiohubImage",
                   return_value=FakeImage()), patch(
                       "linajea.biohub_checkpoint_compare.read_geff_tracks",
                       side_effect=fake_tracks), patch(
                       "linajea.biohub_checkpoint_compare.predict_roi",
                       side_effect=fake_predict) as predict:
            report, summaries = compare_checkpoints(
                root, root, iterations=(1, 2), reuse_iteration=2)
        assert predict.call_count == 1
        assert [row["iteration"] for row in summaries] == [1, 2]
        assert summaries[1]["candidate_hidden_recovered_4um"] == 1
        assert report["reused_iteration"] == 2

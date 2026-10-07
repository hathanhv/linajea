"""Compare cached Biohub training checkpoints on one fixed ROI.

Candidate recovery is the primary diagnostic. Hidden GT is read only after
prediction and fixed-cost ILP selection, and must not be used to tune the
checkpoint or solver in an independent evaluation.
"""

import json
from pathlib import Path

import numpy as np

from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import match_nodes, read_rows, write_rows
from linajea.biohub_reconstruct import (EDGE_FIELDS, NODE_FIELDS,
                                       candidate_edges, predict_roi,
                                       solve_basic_ilp)


THRESHOLDS_UM = (2.0, 4.0, 7.0)
METRIC_FIELDS = ("iteration", "stage", "threshold_um", "node_count",
                 "edge_count", "visible_recovered", "visible_total",
                 "visible_recall", "hidden_recovered", "hidden_total",
                 "hidden_recall", "visible_edges_recovered",
                 "visible_edges_total", "hidden_edges_recovered",
                 "hidden_edges_total", "hidden_edges_detectable")
SUMMARY_FIELDS = ("iteration", "candidate_nodes", "candidate_edges",
                  "candidate_score_mean", "candidate_score_max",
                  "candidate_visible_recovered_4um",
                  "candidate_visible_recall_4um",
                  "candidate_hidden_recovered_4um",
                  "candidate_hidden_recall_4um",
                  "candidate_hidden_edges_4um",
                  "selected_nodes", "selected_edges",
                  "selected_visible_recovered_4um",
                  "selected_hidden_recovered_4um",
                  "selected_hidden_edges_4um")


def _roi_records(records, image, start, end):
    voxel_size = np.asarray(image.voxel_size)
    return [record for record in records
            if np.all((np.asarray([record[k] for k in ("t", "z", "y", "x")],
                                  dtype=float) / voxel_size >= start) &
                      (np.asarray([record[k] for k in ("t", "z", "y", "x")],
                                  dtype=float) / voxel_size < end))]


def _edge_hits(gt_edges, inverse_matches, predicted_edges):
    detectable = {(parent, child) for parent, child in gt_edges
                  if parent in inverse_matches and child in inverse_matches}
    recovered = {(parent, child) for parent, child in detectable
                 if (inverse_matches[parent], inverse_matches[child])
                 in predicted_edges}
    return len(recovered), len(detectable)


def _metrics(iteration, stage, nodes, edges, roi_gt, visible_gt,
             hidden_ids, visible_edges, hidden_edges, threshold_um):
    # Visible coverage uses only visible GT, as in the earlier candidate check.
    visible_matches = match_nodes(nodes, visible_gt, threshold_um)
    inverse_visible = {gt_id: pred_id for pred_id, gt_id in
                       visible_matches.items()}
    # Hidden recovery matches against all GT so one prediction cannot be
    # claimed for a visible and a hidden cell in the same frame.
    all_matches = match_nodes(nodes, roi_gt, threshold_um)
    inverse_all = {gt_id: pred_id for pred_id, gt_id in all_matches.items()}
    hidden_found = len(hidden_ids & set(inverse_all))
    pred_edges = {(int(edge["source_id"]), int(edge["target_id"]))
                  for edge in edges}
    visible_edge_hits, _ = _edge_hits(visible_edges, inverse_visible,
                                     pred_edges)
    hidden_edge_hits, hidden_detectable = _edge_hits(
        hidden_edges, inverse_all, pred_edges)
    return {
        "iteration": iteration, "stage": stage,
        "threshold_um": threshold_um,
        "node_count": len(nodes), "edge_count": len(edges),
        "visible_recovered": len(visible_matches),
        "visible_total": len(visible_gt),
        "visible_recall": len(visible_matches) / len(visible_gt)
        if visible_gt else None,
        "hidden_recovered": hidden_found,
        "hidden_total": len(hidden_ids),
        "hidden_recall": hidden_found / len(hidden_ids)
        if hidden_ids else None,
        "visible_edges_recovered": visible_edge_hits,
        "visible_edges_total": len(visible_edges),
        "hidden_edges_recovered": hidden_edge_hits,
        "hidden_edges_total": len(hidden_edges),
        "hidden_edges_detectable": hidden_detectable,
    }


def compare_checkpoints(data_dir, output_root, iterations=(500, 1000, 1500, 2000),
                        reuse_iteration=2000, device=None):
    """Run missing inference only, then compare candidate and selected graphs."""
    data_dir, output_root = Path(data_dir), Path(output_root)
    manifest_path = output_root / "split.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample = manifest["sample"]
    image_path = data_dir / f"{sample}.zarr"
    config_path = output_root / "train" / "config.toml"
    calibration_path = output_root / "prediction_calibrated" / "calibration.json"
    chosen = json.loads(calibration_path.read_text(encoding="utf-8"))["chosen"]
    comparison_dir = output_root / "checkpoint_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    iterations = tuple(sorted(set(map(int, iterations))))
    if not iterations or reuse_iteration not in iterations:
        raise ValueError("Iteration list must include the cached reuse iteration")
    for iteration in iterations:
        if not (output_root / "train" /
                f"train_net_checkpoint_{iteration}").is_file():
            raise FileNotFoundError(f"Missing checkpoint at iteration {iteration}")

    # First produce candidate and selected graphs without opening original GT.
    graphs = {}
    for iteration in iterations:
        if iteration == reuse_iteration:
            candidate_dir = output_root / "prediction"
            selected_dir = output_root / "prediction_calibrated"
            candidate_nodes = read_rows(candidate_dir / "candidate_nodes.csv")
            candidate_links = read_rows(candidate_dir / "candidate_edges.csv")
            selected_nodes = read_rows(selected_dir / "pred_nodes.csv")
            selected_links = read_rows(selected_dir / "pred_edges.csv")
            print(f"Iteration {iteration}: reusing cached candidates and graph",
                  flush=True)
        else:
            iteration_dir = comparison_dir / f"iteration_{iteration}"
            candidate_file = iteration_dir / "candidate_nodes.csv"
            edge_file = iteration_dir / "candidate_edges.csv"
            if candidate_file.is_file() and edge_file.is_file():
                candidate_nodes = read_rows(candidate_file)
                candidate_links = read_rows(edge_file)
                print(f"Iteration {iteration}: reusing cached inference",
                      flush=True)
            else:
                checkpoint = (output_root / "train" /
                              f"train_net_checkpoint_{iteration}")
                print(f"Iteration {iteration}: running tiled inference",
                      flush=True)
                candidate_nodes, _ = predict_roi(
                    image_path, config_path, checkpoint, manifest_path,
                    iteration_dir, device=device)
                candidate_links = candidate_edges(candidate_nodes)
                write_rows(edge_file, EDGE_FIELDS, candidate_links)
            selected_nodes, selected_links, solver = solve_basic_ilp(
                candidate_nodes, candidate_links,
                selection_constant=float(chosen["selection_constant"]),
                track_cost=float(chosen["track_cost"]))
            write_rows(iteration_dir / "pred_nodes.csv", NODE_FIELDS,
                       selected_nodes)
            write_rows(iteration_dir / "pred_edges.csv", EDGE_FIELDS,
                       selected_links)
            (iteration_dir / "solver.json").write_text(
                json.dumps(solver, indent=2), encoding="utf-8")
        graphs[iteration] = ((candidate_nodes, candidate_links),
                             (selected_nodes, selected_links))

    # Read GT only for the diagnostic table, after inference and fixed-cost ILP.
    image = BiohubImage(image_path)
    full_gt, _ = read_geff_tracks(data_dir / f"{sample}.geff",
                                  image.voxel_size, image.shape)
    visible_gt, _ = read_geff_tracks(output_root / "visible.geff",
                                     image.voxel_size, image.shape)
    start = np.asarray(manifest["roi_start"])
    end = start + np.asarray(manifest["roi_shape"])
    roi_gt = _roi_records(full_gt, image, start, end)
    visible_gt = _roi_records(visible_gt, image, start, end)
    roi_ids = {int(record["cell_id"]) for record in roi_gt}
    visible_ids = {int(record["cell_id"]) for record in visible_gt}
    hidden_ids = set(map(int, manifest["hidden_node_ids"])) & roi_ids
    visible_edges = {(int(record["parent_id"]), int(record["cell_id"]))
                     for record in visible_gt
                     if int(record["parent_id"]) in visible_ids}
    hidden_edges = {tuple(map(int, edge))
                    for edge in manifest["hidden_edges"]
                    if set(edge) <= roi_ids}

    metrics, summaries = [], []
    for iteration in iterations:
        for stage, (nodes, edges) in zip(("candidate", "selected"),
                                          graphs[iteration]):
            for threshold in THRESHOLDS_UM:
                metrics.append(_metrics(
                    iteration, stage, nodes, edges, roi_gt, visible_gt,
                    hidden_ids, visible_edges, hidden_edges, threshold))
        candidate_nodes, candidate_links = graphs[iteration][0]
        selected_nodes, selected_links = graphs[iteration][1]
        candidate_4 = next(row for row in metrics
                           if row["iteration"] == iteration and
                           row["stage"] == "candidate" and
                           row["threshold_um"] == 4.0)
        selected_4 = next(row for row in metrics
                          if row["iteration"] == iteration and
                          row["stage"] == "selected" and
                          row["threshold_um"] == 4.0)
        scores = [float(node["score"]) for node in candidate_nodes]
        summaries.append({
            "iteration": iteration,
            "candidate_nodes": len(candidate_nodes),
            "candidate_edges": len(candidate_links),
            "candidate_score_mean": float(np.mean(scores)) if scores else None,
            "candidate_score_max": float(np.max(scores)) if scores else None,
            "candidate_visible_recovered_4um": candidate_4["visible_recovered"],
            "candidate_visible_recall_4um": candidate_4["visible_recall"],
            "candidate_hidden_recovered_4um": candidate_4["hidden_recovered"],
            "candidate_hidden_recall_4um": candidate_4["hidden_recall"],
            "candidate_hidden_edges_4um":
                candidate_4["hidden_edges_recovered"],
            "selected_nodes": len(selected_nodes),
            "selected_edges": len(selected_links),
            "selected_visible_recovered_4um": selected_4["visible_recovered"],
            "selected_hidden_recovered_4um": selected_4["hidden_recovered"],
            "selected_hidden_edges_4um": selected_4["hidden_edges_recovered"],
        })
    write_rows(comparison_dir / "metrics_all_thresholds.csv", METRIC_FIELDS,
               metrics)
    write_rows(comparison_dir / "summary_4um.csv", SUMMARY_FIELDS, summaries)
    # A diagnostic suggestion from visible GT only; not a generalization
    # estimate, since visible labels were part of training.
    suggested = max(summaries, key=lambda row: (
        row["candidate_visible_recovered_4um"],
        row["selected_visible_recovered_4um"], row["iteration"]))
    report = {
        "sample": sample, "roi_start": start.tolist(),
        "roi_shape": manifest["roi_shape"],
        "iterations": list(iterations),
        "reused_iteration": reuse_iteration,
        "match_thresholds_um": list(THRESHOLDS_UM),
        "fixed_ilp_costs": {"selection_constant":
                             chosen["selection_constant"],
                             "track_cost": chosen["track_cost"],
                             "weight_node_score": 17.0,
                             "edge_weight_per_um": 0.35,
                             "division_cost": 1.0},
        "suggested_iteration_from_visible_only": suggested["iteration"],
        "note": "Exploratory, same-sequence pilot. Hidden GT did not select costs or checkpoint. Visible GT is training data, not independent validation.",
    }
    (comparison_dir / "comparison.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report, summaries

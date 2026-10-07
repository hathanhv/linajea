"""Re-solve cached Biohub candidates using only visible annotations.

This stage never reads the original GEFF, hidden IDs, or hidden edges. It is
an exploratory calibration of ILP costs, not an independent validation run.
"""

import json
from math import ceil
from pathlib import Path

import numpy as np

from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import match_nodes, read_rows, write_rows
from linajea.biohub_reconstruct import EDGE_FIELDS, NODE_FIELDS, solve_basic_ilp


SELECTION_CONSTANTS = (4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)
TRACK_COSTS = (0.0, 1.0, 3.0, 7.0)
MATCH_THRESHOLD_UM = 4.0
MIN_SUPPORTED_RECALL = 0.8


def _visible_roi_records(visible_geff, image_path, manifest_path):
    image = BiohubImage(image_path)
    visible, _ = read_geff_tracks(visible_geff, image.voxel_size, image.shape)
    # Only the ROI coordinates are read from the split manifest. Hidden node
    # IDs and edges in the manifest are deliberately ignored here.
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    start = np.asarray(manifest["roi_start"], dtype=float)
    end = start + np.asarray(manifest["roi_shape"], dtype=float)
    records = []
    for node in visible:
        position = np.asarray([node[k] for k in ("t", "z", "y", "x")],
                              dtype=float) / np.asarray(image.voxel_size)
        if np.all(position >= start) and np.all(position < end):
            records.append(node)
    return records


def _support(nodes, edges, visible):
    matches = match_nodes(nodes, visible, MATCH_THRESHOLD_UM)
    inverse = {gt_id: pred_id for pred_id, gt_id in matches.items()}
    visible_ids = {int(node["cell_id"]) for node in visible}
    visible_edges = {(int(node["parent_id"]), int(node["cell_id"]))
                     for node in visible
                     if int(node["parent_id"]) in visible_ids}
    predicted_edges = {(int(edge["source_id"]), int(edge["target_id"]))
                       for edge in edges}
    found_edges = sum(
        (inverse[parent], inverse[child]) in predicted_edges
        for parent, child in visible_edges
        if parent in inverse and child in inverse)
    return len(matches), found_edges, len(visible_edges)


def calibrate_cached_candidates(image_path, visible_geff, manifest_path,
                                candidate_dir, output_dir):
    """Keep the smallest graph retaining >=80% of visible candidate support.

    Search only the node selection constant and track appearance cost. All
    other costs and candidate extraction remain fixed. Unmatched candidates
    are never classified as negatives because Biohub GT is sparse.
    """
    candidate_dir, output_dir = Path(candidate_dir), Path(output_dir)
    nodes = read_rows(candidate_dir / "candidate_nodes.csv")
    edges = read_rows(candidate_dir / "candidate_edges.csv")
    visible = _visible_roi_records(visible_geff, image_path, manifest_path)
    if not nodes or not visible:
        raise ValueError("Calibration needs candidates and visible GT in ROI")
    supported_nodes, supported_edges, visible_edge_count = _support(
        nodes, edges, visible)
    if not supported_nodes:
        raise ValueError("No candidate matches visible GT at 4 um")
    target_nodes = ceil(MIN_SUPPORTED_RECALL * supported_nodes)
    target_edges = ceil(MIN_SUPPORTED_RECALL * supported_edges)

    trials = []
    for constant in SELECTION_CONSTANTS:
        for appearance in TRACK_COSTS:
            selected_nodes, selected_edges, solver = solve_basic_ilp(
                nodes, edges, time_limit=30,
                selection_constant=constant, track_cost=appearance)
            hit_nodes, hit_edges, _ = _support(selected_nodes, selected_edges,
                                              visible)
            row = {
                "selection_constant": constant,
                "track_cost": appearance,
                "visible_node_recovered": hit_nodes,
                "visible_edge_recovered": hit_edges,
                "selected_nodes": len(selected_nodes),
                "selected_edges": len(selected_edges),
                "solver_status": solver["solver_status"],
                "solver_gap": solver.get("solver_gap"),
            }
            trials.append((row, selected_nodes, selected_edges, solver))
            print("ILP visible calibration:", row, flush=True)

    eligible = [trial for trial in trials
                if trial[0]["visible_node_recovered"] >= target_nodes and
                trial[0]["visible_edge_recovered"] >= target_edges]
    if eligible:
        # Among graphs meeting both visible-support targets, prefer fewer
        # selected cells, then better edge and node coverage.
        chosen = min(eligible, key=lambda trial: (
            trial[0]["selected_nodes"],
            -trial[0]["visible_edge_recovered"],
            -trial[0]["visible_node_recovered"],
            trial[0]["selected_edges"],
            trial[0]["selection_constant"], trial[0]["track_cost"]))
        criterion = "smallest_graph_meeting_visible_support_targets"
    else:
        # No grid point reached the targets: maximize visible coverage,
        # breaking ties toward the smaller graph.
        chosen = max(trials, key=lambda trial: (
            trial[0]["visible_node_recovered"] / supported_nodes +
            (trial[0]["visible_edge_recovered"] / supported_edges
             if supported_edges else 0),
            -trial[0]["selected_nodes"],
            -trial[0]["selected_edges"]))
        criterion = "best_visible_support_available_in_grid"

    row, selected_nodes, selected_edges, solver = chosen
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "selection_grid.csv", tuple(row),
               [trial[0] for trial in trials])
    write_rows(output_dir / "pred_nodes.csv", NODE_FIELDS, selected_nodes)
    write_rows(output_dir / "pred_edges.csv", EDGE_FIELDS, selected_edges)
    report = {
        "purpose": "exploratory_visible_only_ilp_calibration",
        "match_threshold_um": MATCH_THRESHOLD_UM,
        "candidate_nodes": len(nodes),
        "candidate_edges": len(edges),
        "visible_nodes_in_roi": len(visible),
        "visible_edges_in_roi": visible_edge_count,
        "candidate_supported_visible_nodes": supported_nodes,
        "candidate_supported_visible_edges": supported_edges,
        "required_supported_fraction": MIN_SUPPORTED_RECALL,
        "target_visible_nodes": target_nodes,
        "target_visible_edges": target_edges,
        "selection_criterion": criterion,
        "fixed_costs": {"weight_node_score": 17.0,
                        "division_cost": 1.0,
                        "edge_weight_per_um": 0.35},
        "chosen": row,
        "solver": solver,
        "hidden_gt_used_for_selection": False,
    }
    (output_dir / "calibration.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report

"""Read-only edge and centroid diagnostics for Biohub held-out experiments."""

import numpy as np

from linajea.biohub_random_hide import SPACING_UM, match_nodes


PRED_EDGE_FIELDS = (
    "run", "stage", "source_id", "target_id", "source_t", "target_t",
    "source_z", "source_y", "source_x", "target_z", "target_y",
    "target_x", "source_score", "target_score", "movement_residual_um",
    "matched_source_gt", "matched_target_gt", "source_error_um",
    "target_error_um", "gt_relation", "hidden_gt_edge",
    "manual_edge_review", "reviewer_notes",
)
GT_EDGE_FIELDS = (
    "run", "stage", "gt_source_id", "gt_target_id", "gt_relation",
    "source_t", "target_t", "source_z_1over32um", "source_y_1over32um",
    "source_x_1over32um", "target_z_1over32um", "target_y_1over32um",
    "target_x_1over32um", "matched_source_pred",
    "matched_target_pred", "source_error_um", "target_error_um",
    "edge_selected", "failure_reason",
)
LOCALIZATION_FIELDS = (
    "run", "stage", "match_radius_um", "visible_matched", "visible_total",
    "hidden_matched", "hidden_total", "median_error_um", "p90_error_um",
)


def _error_um(prediction, gt):
    pred_xyz = np.array([float(prediction[axis]) for axis in ("z", "y", "x")])
    gt_xyz = np.array([float(gt[axis]) for axis in ("z", "y", "x")])
    return float(np.linalg.norm(pred_xyz * SPACING_UM - gt_xyz / 32))


def audit_temporal_edges(nodes, edges, full_gt, visible_gt, manifest,
                         run_label, stage, radius_um=4.0):
    """Return predicted-edge, known-GT-edge, and localization audit rows.

    Unknown edges are not called false positives: the original GEFF is sparse.
    The original GEFF is used here only for evaluation, after graph selection.
    """
    pred_by_id = {int(node["node_id"]): node for node in nodes}
    gt_by_id = {int(node["cell_id"]): node for node in full_gt}
    visible_ids = {int(node["cell_id"]) for node in visible_gt}
    hidden_ids = set(map(int, manifest["hidden_node_ids"])) & set(gt_by_id)
    hidden_edges = {tuple(map(int, pair)) for pair in manifest["hidden_edges"]}
    known_edges = {(int(node["parent_id"]), int(node["cell_id"]))
                   for node in full_gt
                   if int(node["parent_id"]) in gt_by_id}
    selected_edges = {(int(edge["source_id"]), int(edge["target_id"]))
                      for edge in edges}
    matches = match_nodes(nodes, full_gt, radius_um)
    inverse = {gt_id: pred_id for pred_id, gt_id in matches.items()}

    pred_rows = []
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = pred_by_id[source_id]
        target = pred_by_id[target_id]
        source_gt_id = matches.get(source_id)
        target_gt_id = matches.get(target_id)
        pair = (source_gt_id, target_gt_id)
        if None in pair:
            relation = "unmatched_endpoint"
        elif pair in hidden_edges and pair in known_edges:
            relation = "confirmed_hidden_edge"
        elif pair in known_edges:
            relation = "confirmed_visible_edge"
        elif int(gt_by_id[target_gt_id]["parent_id"]) in gt_by_id:
            relation = "known_parent_disagrees"
        else:
            relation = "unverified_relation"
        pred_rows.append({
            "run": run_label, "stage": stage,
            "source_id": source_id, "target_id": target_id,
            **{f"source_{axis}": int(source[axis]) for axis in ("t", "z", "y", "x")},
            **{f"target_{axis}": int(target[axis]) for axis in ("t", "z", "y", "x")},
            "source_score": float(source["score"]),
            "target_score": float(target["score"]),
            "movement_residual_um": float(edge["prediction_distance_um"]),
            "matched_source_gt": source_gt_id,
            "matched_target_gt": target_gt_id,
            "source_error_um": _error_um(source, gt_by_id[source_gt_id])
            if source_gt_id is not None else None,
            "target_error_um": _error_um(target, gt_by_id[target_gt_id])
            if target_gt_id is not None else None,
            "gt_relation": relation,
            "hidden_gt_edge": pair in hidden_edges and pair in known_edges,
            "manual_edge_review": "", "reviewer_notes": "",
        })

    gt_rows = []
    for parent_id, child_id in sorted(known_edges):
        parent, child = gt_by_id[parent_id], gt_by_id[child_id]
        source_pred_id = inverse.get(parent_id)
        target_pred_id = inverse.get(child_id)
        selected = ((source_pred_id, target_pred_id) in selected_edges
                    if source_pred_id is not None and target_pred_id is not None
                    else False)
        if selected:
            failure = "recovered"
        elif source_pred_id is None and target_pred_id is None:
            failure = "both_centers_missing"
        elif source_pred_id is None:
            failure = "source_center_missing"
        elif target_pred_id is None:
            failure = "target_center_missing"
        else:
            failure = "edge_missing_despite_both_centers"
        gt_rows.append({
            "run": run_label, "stage": stage,
            "gt_source_id": parent_id, "gt_target_id": child_id,
            "gt_relation": "hidden" if (parent_id, child_id) in hidden_edges
            else "visible",
            "source_t": int(parent["t"]), "target_t": int(child["t"]),
            **{f"source_{axis}_1over32um": int(parent[axis])
               for axis in ("z", "y", "x")},
            **{f"target_{axis}_1over32um": int(child[axis])
               for axis in ("z", "y", "x")},
            "matched_source_pred": source_pred_id,
            "matched_target_pred": target_pred_id,
            "source_error_um": _error_um(pred_by_id[source_pred_id], parent)
            if source_pred_id is not None else None,
            "target_error_um": _error_um(pred_by_id[target_pred_id], child)
            if target_pred_id is not None else None,
            "edge_selected": selected, "failure_reason": failure,
        })

    localization_rows = []
    for radius in (2.0, 4.0, 7.0, 10.0):
        radius_matches = match_nodes(nodes, full_gt, radius)
        matched_gt = set(radius_matches.values())
        errors = [_error_um(pred_by_id[pred_id], gt_by_id[gt_id])
                  for pred_id, gt_id in radius_matches.items()]
        localization_rows.append({
            "run": run_label, "stage": stage, "match_radius_um": radius,
            "visible_matched": len(matched_gt & visible_ids),
            "visible_total": len(visible_ids),
            "hidden_matched": len(matched_gt & hidden_ids),
            "hidden_total": len(hidden_ids),
            "median_error_um": float(np.median(errors)) if errors else None,
            "p90_error_um": float(np.percentile(errors, 90)) if errors else None,
        })
    return pred_rows, gt_rows, localization_rows

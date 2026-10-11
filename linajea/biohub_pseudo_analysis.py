"""Cached-candidate diagnostics for cautious Biohub pseudo-label selection.

All graph selection and filtering uses cached predictions plus visible GT.
Original GEFF is opened only after the policy graphs have been saved, for
held-out diagnostic metrics. Nothing in this module retrains a network.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import zarr

from linajea.biohub_full_export import solve_components
from linajea.biohub_io import read_geff_tracks
from linajea.biohub_random_hide import match_nodes, read_rows, write_rows
from linajea.biohub_reconstruct import EDGE_FIELDS, NODE_FIELDS


POLICIES = (
    ("anchor_loose", 0.20, 12.0, 5),
    ("anchor_mid", 0.30, 8.0, 4),
    ("anchor_strict", 0.40, 4.0, 3),
    ("anchor_very_strict", 0.50, 3.0, 2),
)
DEFAULT_POLICY = "anchor_strict"
MATCH_UM = 4.0


def discover_runs(source_dir):
    """Accept the multiseed notebook output or the older checkpoint notebook."""
    source_dir = Path(source_dir)
    seed_dirs = sorted(path for path in source_dir.glob("seed_*")
                       if path.is_dir())
    if seed_dirs:
        runs = {path.name: (path, path / "prediction") for path in seed_dirs}
    elif (source_dir / "checkpoint_comparison" / "iteration_1500" /
          "candidate_nodes.csv").is_file():
        runs = {"checkpoint_1500":
                (source_dir, source_dir / "checkpoint_comparison" /
                 "iteration_1500")}
    elif (source_dir / "prediction" / "candidate_nodes.csv").is_file():
        runs = {"single_run": (source_dir, source_dir / "prediction")}
    else:
        raise FileNotFoundError(
            f"No cached Biohub candidates under {source_dir}. Expected "
            "seed_*/prediction/ or checkpoint_comparison/iteration_1500/.")
    for label, (root, prediction) in runs.items():
        for path in (root / "split.json", root / "visible.geff",
                     prediction / "candidate_nodes.csv",
                     prediction / "candidate_edges.csv"):
            if not path.exists():
                raise FileNotFoundError(f"{label} is missing {path}")
    return runs


def _visible_records(geff_path, manifest):
    group = zarr.open_group(str(geff_path), mode="r")
    axes = group.attrs["geff"]["axes"]
    voxel_size = (1,) + tuple(int(round(float(axis["scale"]) * 32))
                              for axis in axes[1:])
    records, _ = read_geff_tracks(geff_path, voxel_size)
    start = np.asarray(manifest["roi_start"])
    end = start + np.asarray(manifest["roi_shape"])
    scale = np.asarray(voxel_size)
    records = [record for record in records
               if np.all(np.asarray([record[axis] for axis in
                                     ("t", "z", "y", "x")]) / scale >= start)
               and np.all(np.asarray([record[axis] for axis in
                                      ("t", "z", "y", "x")]) / scale < end)]
    return records, voxel_size


def candidate_frame_diagnostics(nodes, first_frame, frame_count,
                                candidate_cap=200):
    by_frame = defaultdict(list)
    for node in nodes:
        by_frame[int(node["t"])].append(float(node["score"]))
    rows = []
    for t in range(first_frame, first_frame + frame_count):
        scores = np.asarray(by_frame[t], dtype=float)
        rows.append({"t": t, "candidate_count": len(scores),
                     "at_candidate_cap": len(scores) >= candidate_cap,
                     "score_p10": float(np.percentile(scores, 10))
                     if len(scores) else None,
                     "score_median": float(np.median(scores))
                     if len(scores) else None,
                     "score_p90": float(np.percentile(scores, 90))
                     if len(scores) else None,
                     "score_max": float(np.max(scores))
                     if len(scores) else None})
    return rows


def anchored_corridor(nodes, edges, visible_records, min_score=0.4,
                      max_edge_distance_um=4.0, max_missing_frames=3,
                      match_um=MATCH_UM):
    """Keep only novel nodes on selected paths between two visible anchors.

    Visible anchors are matched to selected predictions; this deliberately
    does not insert or force GT nodes into the ILP. It is a conservative
    post-filter, not a full GT-constrained reconstruction solver.
    """
    matches = match_nodes(nodes, visible_records, match_um)
    anchors = set(matches)
    by_id = {int(node["node_id"]): node for node in nodes}
    allowed = {node_id for node_id, node in by_id.items()
               if node_id in anchors or float(node["score"]) >= min_score}
    links = [(int(edge["source_id"]), int(edge["target_id"]), edge)
             for edge in edges
             if int(edge["source_id"]) in allowed
             and int(edge["target_id"]) in allowed
             and float(edge["prediction_distance_um"]) <= max_edge_distance_um]
    parents, children = defaultdict(list), defaultdict(list)
    for parent, child, _ in links:
        if int(by_id[child]["t"]) - int(by_id[parent]["t"]) != 1:
            raise ValueError("Anchor corridor needs adjacent-frame edges")
        parents[child].append(parent)
        children[parent].append(child)

    order = sorted(allowed, key=lambda node_id: int(by_id[node_id]["t"]))
    far = float("inf")
    from_left, from_right = {}, {}
    for node_id in order:
        from_left[node_id] = (0 if node_id in anchors else
                              min((from_left[parent] + 1
                                   for parent in parents[node_id]),
                                  default=far))
    for node_id in reversed(order):
        from_right[node_id] = (0 if node_id in anchors else
                               min((from_right[child] + 1
                                    for child in children[node_id]),
                                   default=far))
    pseudo_ids = {node_id for node_id in allowed - anchors
                  if from_left[node_id] < far and from_right[node_id] < far
                  and from_left[node_id] + from_right[node_id] - 1
                  <= max_missing_frames}
    kept_ids = anchors | pseudo_ids
    kept_nodes = [node for node in nodes
                  if int(node["node_id"]) in kept_ids]
    kept_edges = [edge for parent, child, edge in links
                  if parent in kept_ids and child in kept_ids
                  and from_left[parent] < far and from_right[child] < far
                  and from_left[parent] + from_right[child]
                  <= max_missing_frames]
    pseudo_nodes = [node for node in kept_nodes
                    if int(node["node_id"]) in pseudo_ids]
    return kept_nodes, kept_edges, pseudo_nodes, {
        "matched_visible_anchors": len(anchors),
        "two_sided_pseudo_nodes": len(pseudo_nodes),
        "anchor_matches": matches,
    }


def _graph_metrics(nodes, edges, visible_gt, full_gt, manifest):
    visible_matches = match_nodes(nodes, visible_gt, MATCH_UM)
    visible_inverse = {gt: pred for pred, gt in visible_matches.items()}
    visible_ids = {int(node["cell_id"]) for node in visible_gt}
    visible_edges = {(int(node["parent_id"]), int(node["cell_id"]))
                     for node in visible_gt
                     if int(node["parent_id"]) in visible_ids}
    selected_edges = {(int(edge["source_id"]), int(edge["target_id"]))
                      for edge in edges}
    visible_edge_recovered = sum(
        (visible_inverse[parent], visible_inverse[child]) in selected_edges
        for parent, child in visible_edges
        if parent in visible_inverse and child in visible_inverse)
    result = {
        "visible_recovered": len(visible_matches),
        "visible_total": len(visible_gt),
        "visible_edge_recovered": visible_edge_recovered,
        "visible_edge_total": len(visible_edges),
        "new_not_matched_visible": len(nodes) - len(visible_matches),
        "hidden_recovered": None, "hidden_total": None,
        "hidden_edge_recovered": None, "hidden_edge_total": None,
        "hidden_edge_endpoint_detectable": None,
        "unconfirmed_after_full_gt": None,
        "hidden_confirmation_lower_bound": None,
    }
    if full_gt is None:
        return result
    matched = match_nodes(nodes, full_gt, MATCH_UM)
    inverse = {gt: pred for pred, gt in matched.items()}
    full_ids = {int(node["cell_id"]) for node in full_gt}
    hidden_ids = set(map(int, manifest["hidden_node_ids"])) & full_ids
    hidden_edges = {tuple(map(int, edge))
                    for edge in manifest["hidden_edges"]
                    if set(map(int, edge)) <= full_ids}
    recovered_hidden = len(hidden_ids & set(inverse))
    detectable = {edge for edge in hidden_edges
                  if edge[0] in inverse and edge[1] in inverse}
    recovered_edges = sum((inverse[parent], inverse[child]) in selected_edges
                          for parent, child in detectable)
    novel = {int(node["node_id"]) for node in nodes} - {
        pred for pred, gt in matched.items() if gt in visible_ids}
    result.update({
        "hidden_recovered": recovered_hidden,
        "hidden_total": len(hidden_ids),
        "hidden_edge_recovered": recovered_edges,
        "hidden_edge_total": len(hidden_edges),
        "hidden_edge_endpoint_detectable": len(detectable),
        "new_not_matched_visible": len(novel),
        "unconfirmed_after_full_gt": len(novel) - recovered_hidden,
        "hidden_confirmation_lower_bound":
            recovered_hidden / len(novel) if novel else None,
    })
    return result


def _audit_rows(selected_nodes, policy_nodes, visible_gt, run_label,
                sample_per_group=50):
    matched = match_nodes(selected_nodes, visible_gt, MATCH_UM)
    pseudo_ids = {int(node["node_id"]) for node in policy_nodes}
    groups = {
        "anchored_pseudo": [node for node in selected_nodes
                            if int(node["node_id"]) in pseudo_ids],
        "other_new_prediction": [node for node in selected_nodes
                                 if int(node["node_id"]) not in pseudo_ids
                                 and int(node["node_id"]) not in matched],
    }
    rng = np.random.default_rng(20261011)
    rows = []
    for group, candidates in groups.items():
        if not candidates:
            continue
        indices = rng.choice(len(candidates),
                             size=min(sample_per_group, len(candidates)),
                             replace=False)
        for index in indices:
            node = candidates[int(index)]
            rows.append({"run": run_label, "group": group,
                         "node_id": int(node["node_id"]),
                         "t": int(node["t"]), "z": int(node["z"]),
                         "y": int(node["y"]), "x": int(node["x"]),
                         "score": float(node["score"]),
                         "manual_cell_review": "", "reviewer_notes": ""})
    return rows


def analyze_cached(source_dir, output_dir, gt_dir=None,
                   selection_constant=7.0, track_cost=7.0,
                   time_limit_per_component=300):
    """Run fixed-cost ILP and anchor filtering without retraining/inference."""
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    runs = discover_runs(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    frame_rows, audit_rows = [], []

    # Selection and pseudo-label filtering do not open original/hidden GT.
    for label, (run_root, candidate_dir) in runs.items():
        manifest = json.loads((run_root / "split.json").read_text(encoding="utf-8"))
        visible_gt, voxel_size = _visible_records(run_root / "visible.geff",
                                                   manifest)
        candidates = read_rows(candidate_dir / "candidate_nodes.csv")
        candidate_edges = read_rows(candidate_dir / "candidate_edges.csv")
        first_frame = int(manifest["roi_start"][0])
        diagnostics = candidate_frame_diagnostics(
            candidates, first_frame, int(manifest["roi_shape"][0]))
        frame_rows.extend({"run": label, **row} for row in diagnostics)
        print(label, "candidates:", len(candidates), "cap frames:",
              sum(row["at_candidate_cap"] for row in diagnostics), flush=True)

        selected, selected_edges, statuses = solve_components(
            candidates, candidate_edges, selection_constant, track_cost,
            time_limit_per_component=time_limit_per_component,
            first_frame=first_frame)
        run_out = output_dir / label
        fixed_out = run_out / "fixed_ilp"
        fixed_out.mkdir(parents=True, exist_ok=True)
        write_rows(fixed_out / "pred_nodes.csv", NODE_FIELDS, selected)
        write_rows(fixed_out / "pred_edges.csv", EDGE_FIELDS, selected_edges)
        solver_info = {"selection_constant": selection_constant,
                       "track_cost": track_cost,
                       "component_count": len(statuses),
                       "solver_status_counts": dict(Counter(
                           status["solver_status"] for status in statuses)),
                       "max_solver_gap": max((status.get("solver_gap") or 0
                                              for status in statuses), default=0)}
        nonoptimal = {message for message in solver_info["solver_status_counts"]
                      if message != "analytical_singleton" and
                      "Optimal" not in message}
        if nonoptimal:
            print(label, "WARNING: some ILP components are not optimal:",
                  sorted(nonoptimal), flush=True)
        (fixed_out / "solver.json").write_text(
            json.dumps(solver_info, indent=2), encoding="utf-8")
        policies = {}
        for policy_name, min_score, max_edge_um, max_missing in POLICIES:
            graph_nodes, graph_edges, pseudo_nodes, policy_info = anchored_corridor(
                selected, selected_edges, visible_gt, min_score, max_edge_um,
                max_missing)
            policies[policy_name] = (graph_nodes, graph_edges,
                                     pseudo_nodes, policy_info)
            if policy_name == DEFAULT_POLICY:
                policy_out = run_out / "anchored_strict"
                policy_out.mkdir(parents=True, exist_ok=True)
                write_rows(policy_out / "pred_nodes.csv", NODE_FIELDS,
                           graph_nodes)
                write_rows(policy_out / "pred_edges.csv", EDGE_FIELDS,
                           graph_edges)
                write_rows(policy_out / "pseudo_nodes_only.csv", NODE_FIELDS,
                           pseudo_nodes)
                write_rows(policy_out / "anchor_matches.csv",
                           ("predicted_node_id", "visible_cell_id"),
                           [{"predicted_node_id": pred,
                             "visible_cell_id": gt}
                            for pred, gt in sorted(
                                policy_info["anchor_matches"].items())])
                audit_rows.extend(_audit_rows(selected, pseudo_nodes,
                                              visible_gt, label))
        outputs[label] = {"manifest": manifest, "voxel_size": voxel_size,
                          "visible_gt": visible_gt,
                          "graphs": {"candidate":
                                     (candidates, candidate_edges, None),
                                     "fixed_ilp":
                                     (selected, selected_edges, None),
                                     **policies}}
        print(label, "fixed ILP selected:", len(selected),
              "strict two-anchor pseudo:",
              len(policies[DEFAULT_POLICY][2]), flush=True)

    if frame_rows:
        write_rows(output_dir / "candidate_frames.csv", tuple(frame_rows[0]),
                   frame_rows)
    write_rows(output_dir / "audit_queue.csv",
               ("run", "group", "node_id", "t", "z", "y", "x", "score",
                "manual_cell_review", "reviewer_notes"), audit_rows)

    # Held-out GT enters only here, after all policies and graphs are fixed.
    summary = []
    for label, payload in outputs.items():
        manifest = payload["manifest"]
        full_gt = None
        if gt_dir is not None:
            original = Path(gt_dir) / f"{manifest['sample']}.geff"
            if not original.is_dir():
                raise FileNotFoundError(f"Original GEFF not found: {original}")
            full_records, _ = read_geff_tracks(original,
                                               payload["voxel_size"])
            start = np.asarray(manifest["roi_start"])
            end = start + np.asarray(manifest["roi_shape"])
            scale = np.asarray(payload["voxel_size"])
            full_gt = [record for record in full_records
                       if np.all(np.asarray([record[axis] for axis in
                                             ("t", "z", "y", "x")]) /
                                 scale >= start)
                       and np.all(np.asarray([record[axis] for axis in
                                              ("t", "z", "y", "x")]) /
                                  scale < end)]
        for stage, graph in payload["graphs"].items():
            if stage in ("candidate", "fixed_ilp"):
                nodes, edges, _ = graph
                pseudo_count = None
                min_score = max_edge_um = max_missing = None
                anchor_count = None
            else:
                nodes, edges, pseudo, info = graph
                pseudo_count = len(pseudo)
                min_score, max_edge_um, max_missing = next(
                    (score, edge_um, gap) for name, score, edge_um, gap
                    in POLICIES if name == stage)
                anchor_count = info["matched_visible_anchors"]
            row = {"run": label, "stage": stage,
                   "node_count": len(nodes), "edge_count": len(edges),
                   "anchor_count": anchor_count,
                   "pseudo_node_count": pseudo_count,
                   "min_score": min_score,
                   "max_edge_distance_um": max_edge_um,
                   "max_missing_frames": max_missing,
                   **_graph_metrics(nodes, edges, payload["visible_gt"],
                                    full_gt, manifest)}
            summary.append(row)
    write_rows(output_dir / "summary.csv", tuple(summary[0]), summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2),
                                             encoding="utf-8")
    settings = {"source_dir": str(source_dir),
                "gt_dir": str(gt_dir) if gt_dir is not None else None,
                "fixed_ilp": {"selection_constant": selection_constant,
                              "track_cost": track_cost,
                              "weight_node_score": 17.0,
                              "edge_weight_per_um": 0.35,
                              "division_cost": 1.0},
                "match_distance_um": MATCH_UM,
                "policies": [{"name": name, "min_score": score,
                              "max_edge_distance_um": edge,
                              "max_missing_frames": gap}
                             for name, score, edge, gap in POLICIES],
                "policy_selection_used_hidden_gt": False,
                "note": "Exploratory same-sequence diagnostic; unmatched predictions are unknown."}
    (output_dir / "settings.json").write_text(json.dumps(settings, indent=2),
                                              encoding="utf-8")
    return summary

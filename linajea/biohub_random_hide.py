"""Split and evaluate a held-out Biohub GEFF annotation experiment.

Only create_split and evaluate read the original GEFF. Training and prediction
consume the filtered GEFF and raw image, respectively.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from linajea.biohub_io import BiohubImage, read_geff_tracks


AXES = ("t", "z", "y", "x")
SPACING_UM = np.array((1.625, 0.40625, 0.40625))


def _components(ids, positions_um, times, min_distance_um):
    parent = {int(node): int(node) for node in ids}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for t in np.unique(times):
        indices = np.flatnonzero(times == t)
        if len(indices) < 2:
            continue
        pairs = cKDTree(positions_um[indices]).query_pairs(min_distance_um)
        for a, b in pairs:
            ra, rb = root(int(ids[indices[a]])), root(int(ids[indices[b]]))
            parent[ra] = rb
    groups = {}
    for node in ids:
        node = int(node)
        groups.setdefault(root(node), []).append(node)
    return list(groups.values())


def _select_roi(voxels, image_shape, roi_shape):
    """Choose a fixed, annotation-rich pilot ROI before random hiding."""
    size = np.minimum(np.asarray(roi_shape), np.asarray(image_shape))
    best_count, best_start = -1, None
    for point in voxels:
        start = np.clip(point - size // 2, 0, np.asarray(image_shape) - size)
        count = int(np.all((voxels >= start) & (voxels < start + size), axis=1).sum())
        if count > best_count:
            best_count, best_start = count, start
    return [int(v) for v in best_start], [int(v) for v in size], best_count


def create_split(data_dir, sample, output_dir, hide_rate=0.2, seed=42,
                 roi_shape=(20, 32, 128, 128), mask_radius_um=2.0):
    import zarr

    data_dir, output_dir = Path(data_dir), Path(output_dir)
    image = BiohubImage(data_dir / f"{sample}.zarr")
    original = data_dir / f"{sample}.geff"
    read_geff_tracks(original, image.voxel_size, image.shape)
    group = zarr.open_group(str(original), mode="r")
    ids = np.asarray(group["nodes/ids"])
    positions = np.stack([np.asarray(group[f"nodes/props/{axis}/values"])
                          for axis in AXES], axis=1)
    edges = np.asarray(group["edges/ids"])
    voxels = np.rint(positions).astype(int)
    roi_start, roi_size, roi_gt_count = _select_roi(voxels, image.shape, roi_shape)
    components = _components(ids, positions[:, 1:] * SPACING_UM,
                             voxels[:, 0], mask_radius_um)
    rng = np.random.default_rng(seed)
    hidden = set()
    target = round(len(ids) * hide_rate)
    for index in rng.permutation(len(components)):
        component = components[int(index)]
        if abs(target - (len(hidden) + len(component))) < abs(target - len(hidden)):
            hidden.update(component)
    visible = set(map(int, ids)) - hidden
    visible_mask = np.array([int(node) in visible for node in ids])
    edge_mask = np.array([int(a) in visible and int(b) in visible
                          for a, b in edges])
    if not visible or not edge_mask.any():
        raise ValueError("The split has no visible nodes or temporal edges")
    # The component split guarantees that hidden centers are not in a visible
    # annotation's spatial loss mask at the same time point.
    for t in np.unique(voxels[:, 0]):
        h = positions[(voxels[:, 0] == t) & ~visible_mask, 1:] * SPACING_UM
        v = positions[(voxels[:, 0] == t) & visible_mask, 1:] * SPACING_UM
        if len(h) and len(v) and cKDTree(v).query(h)[0].min() <= mask_radius_um:
            raise AssertionError("Hidden center overlaps a visible loss mask")

    output_dir.mkdir(parents=True, exist_ok=True)
    visible_path = output_dir / "visible.geff"
    out = zarr.open_group(str(visible_path), mode="w")
    out.attrs["geff"] = dict(group.attrs["geff"])
    out.create_group("nodes")
    out.create_group("edges")
    out["nodes"].create_array("ids", data=ids[visible_mask])
    out["edges"].create_array("ids", data=edges[edge_mask])
    props = out["nodes"].create_group("props")
    for axis, values in zip(AXES, positions.T):
        props.create_group(axis).create_array("values", data=values[visible_mask])
    records, edge_count = read_geff_tracks(visible_path, image.voxel_size,
                                           image.shape)
    assert len(records) == visible_mask.sum() and edge_count == edge_mask.sum()
    assert {int(r["cell_id"]) for r in records} == visible
    assert not any(int(a) in hidden or int(b) in hidden
                   for a, b in np.asarray(out["edges/ids"]))
    manifest = {
        "sample": sample, "seed": seed, "requested_hide_rate": hide_rate,
        "actual_hide_rate": len(hidden) / len(ids),
        "mask_radius_um": mask_radius_um,
        "visible_node_ids": sorted(visible), "hidden_node_ids": sorted(hidden),
        "visible_edges": edges[edge_mask].astype(int).tolist(),
        "hidden_edges": edges[~edge_mask].astype(int).tolist(),
        "roi_start": roi_start, "roi_shape": roi_size,
        "roi_gt_count": roi_gt_count,
        "original_node_count": len(ids), "original_edge_count": len(edges),
    }
    (output_dir / "split.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    return manifest


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def match_nodes(predicted, gt_records, threshold_um):
    """Maximum-cardinality, minimum-distance matching within each frame."""
    pred_to_gt = {}
    for t in sorted({int(record["t"]) for record in gt_records}):
        p_indices = [i for i, p in enumerate(predicted) if int(p["t"]) == t]
        g_indices = [i for i, g in enumerate(gt_records) if int(g["t"]) == t]
        if not p_indices or not g_indices:
            continue
        pred_xyz = np.array([[float(predicted[i][d]) for d in ("z", "y", "x")]
                             for i in p_indices]) * SPACING_UM
        gt_xyz = np.array([[float(gt_records[i][d]) for d in ("z", "y", "x")]
                           for i in g_indices]) / 32  # records are in 1/32 um
        distances = np.linalg.norm(pred_xyz[:, None] - gt_xyz[None, :], axis=2)
        # A dummy assignment must cost more than every possible gain from
        # rearranging valid matches, so cardinality wins before distance.
        dummy = (min(len(p_indices), len(g_indices)) + 1) * (threshold_um + 1)
        large = 2 * dummy
        costs = np.where(distances <= threshold_um, distances, large)
        # Dummy columns allow unmatched predictions; valid matches always win.
        costs = np.concatenate([costs,
                                np.full((len(p_indices), len(p_indices)),
                                        dummy)], axis=1)
        row, col = linear_sum_assignment(costs)
        for a, b in zip(row, col):
            if b < len(g_indices) and distances[a, b] <= threshold_um:
                pred_to_gt[int(predicted[p_indices[a]]["node_id"])] = int(
                    gt_records[g_indices[b]]["cell_id"])
    return pred_to_gt


def evaluate(data_dir, manifest_path, nodes_path, edges_path, output_dir,
             thresholds=(2.0, 4.0, 7.0)):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sample = manifest["sample"]
    image = BiohubImage(Path(data_dir) / f"{sample}.zarr")
    gt, _ = read_geff_tracks(Path(data_dir) / f"{sample}.geff",
                             image.voxel_size, image.shape)
    start = np.asarray(manifest["roi_start"])
    end = start + np.asarray(manifest["roi_shape"])
    roi_gt = [r for r in gt if np.all((np.array([r[d] for d in AXES]) /
                                     np.asarray(image.voxel_size) >= start) &
                                    (np.array([r[d] for d in AXES]) /
                                     np.asarray(image.voxel_size) < end))]
    gt_ids = {int(r["cell_id"]) for r in roi_gt}
    hidden_ids = set(manifest["hidden_node_ids"]) & gt_ids
    visible_ids = set(manifest["visible_node_ids"]) & gt_ids
    hidden_edges = {tuple(edge) for edge in manifest["hidden_edges"]
                    if set(edge) <= gt_ids}
    predicted = read_rows(nodes_path)
    pred_edges = {(int(row["source_id"]), int(row["target_id"]))
                  for row in read_rows(edges_path)}
    results = {}
    for threshold in thresholds:
        matched = match_nodes(predicted, roi_gt, threshold)
        inverse = {gt_id: pred_id for pred_id, gt_id in matched.items()}
        recovered_hidden = hidden_ids & set(inverse)
        recovered_edges = {edge for edge in hidden_edges
                           if edge[0] in inverse and edge[1] in inverse and
                           (inverse[edge[0]], inverse[edge[1]]) in pred_edges}
        detectable = {edge for edge in hidden_edges
                      if edge[0] in inverse and edge[1] in inverse}
        novel = {int(row["node_id"]) for row in predicted} - {
            pred_id for pred_id, gt_id in matched.items() if gt_id in visible_ids}
        results[str(threshold)] = {
            "hidden_node_count": len(hidden_ids),
            "hidden_node_recovered": len(recovered_hidden),
            "hidden_node_recall": len(recovered_hidden) / len(hidden_ids)
            if hidden_ids else None,
            "hidden_edge_count": len(hidden_edges),
            "hidden_edge_recovered": len(recovered_edges),
            "hidden_edge_recall": len(recovered_edges) / len(hidden_edges)
            if hidden_edges else None,
            "hidden_edge_endpoint_detectable": len(detectable),
            "conditional_edge_recall": len(recovered_edges) / len(detectable)
            if detectable else None,
            "new_prediction_count": len(novel),
            "hidden_confirmed_new_predictions": len(recovered_hidden),
            "hidden_confirmation_rate_lower_bound": len(recovered_hidden) / len(novel)
            if novel else None,
        }
    summary = {"sample": sample, "roi_start": start.tolist(),
               "roi_shape": manifest["roi_shape"],
               "predicted_nodes": len(predicted),
               "predicted_edges": len(pred_edges), "thresholds_um": results,
               "precision_note": "Unmatched predictions are unknown because Biohub GT is sparse."}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2),
                                              encoding="utf-8")
    return summary

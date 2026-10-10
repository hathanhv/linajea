"""Small-ROI, database-free Linajea inference and basic ILP reconstruction."""

import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from linajea.biohub_io import BiohubImage
from linajea.biohub_random_hide import SPACING_UM, write_rows


NODE_FIELDS = ("node_id", "t", "z", "y", "x", "score", "mv_z_um", "mv_y_um", "mv_x_um")
EDGE_FIELDS = ("source_id", "target_id", "prediction_distance_um")


def _starts(length, step):
    return range(0, length, step)


def predict_roi(image_path, config_path, checkpoint_path, manifest_path,
                output_dir, score_threshold=0.2, max_candidates_per_frame=200,
                device=None):
    import torch
    from linajea.config import TrackingConfig
    from linajea.training.torch_model import UnetModelWrapper

    image = BiohubImage(image_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    start = np.asarray(manifest["roi_start"], dtype=int)
    size = np.asarray(manifest["roi_shape"], dtype=int)
    config = TrackingConfig.from_file(str(config_path))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = UnetModelWrapper(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    input_shape = np.asarray(config.model.predict_input_shape, dtype=int)
    with torch.no_grad():
        trial = model(torch.zeros(tuple(input_shape), device=device))
    output_shape = np.asarray(trial[0].shape, dtype=int)
    if output_shape[0] != 1 or output_shape[1:].min() < 1:
        raise ValueError(f"Unexpected model output shape {output_shape}")
    step = output_shape[1:]
    halo = (input_shape[1:] - step) // 2
    t_halo = (input_shape[0] - 1) // 2
    if np.any((input_shape[1:] - step) % 2) or input_shape[0] != 2 * t_halo + 1:
        raise ValueError("Model input/output margins are not symmetric")
    qlo, qhi = image.attrs["stats"]["q001"], image.attrs["stats"]["q099"]
    if qhi <= qlo:
        raise ValueError("Invalid normalization quantiles")

    @lru_cache(maxsize=8)
    def frame(t):
        return image.read((slice(t, t + 1), slice(None), slice(None),
                           slice(None)))[0]

    ci = np.zeros(tuple(size), dtype=np.float32)
    mv = np.zeros((3,) + tuple(size), dtype=np.float32)
    coverage = np.zeros(tuple(size), dtype=np.uint8)
    with torch.no_grad():
        for local_t in range(size[0]):
            t = int(start[0] + local_t)
            for local_z, local_y, local_x in product(
                    _starts(size[1], step[0]), _starts(size[2], step[1]),
                    _starts(size[3], step[2])):
                out_local = np.array((local_z, local_y, local_x), dtype=int)
                out_global = start[1:] + out_local
                read_begin = out_global - halo
                read_end = read_begin + input_shape[1:]
                valid_begin = np.maximum(read_begin, 0)
                valid_end = np.minimum(read_end, image.shape[1:])
                dest_begin = valid_begin - read_begin
                dest_end = dest_begin + valid_end - valid_begin
                raw = np.zeros(tuple(input_shape), dtype=np.float32)
                if np.all(valid_end > valid_begin):
                    dst = tuple(slice(int(a), int(b)) for a, b in
                                zip(dest_begin, dest_end))
                    src = tuple(slice(int(a), int(b)) for a, b in
                                zip(valid_begin, valid_end))
                    for dt in range(input_shape[0]):
                        ft = t + dt - t_halo
                        if 0 <= ft < image.shape[0]:
                            values = frame(ft)[src].astype(np.float32)
                            raw[(dt,) + dst] = (
                                np.clip(values, qlo / 2, qhi * 2) - qlo
                            ) / (qhi - qlo)
                tensor = torch.from_numpy(raw).to(device)
                result = model(tensor)
                block_ci = result[0][0].detach().cpu().numpy()
                block_mv = result[3][:, 0].detach().cpu().numpy()
                lengths = np.minimum(step, size[1:] - out_local)
                dst = tuple(slice(int(a), int(a + n))
                            for a, n in zip(out_local, lengths))
                src = tuple(slice(0, int(n)) for n in lengths)
                ci[(local_t,) + dst] = block_ci[src]
                mv[(slice(None), local_t) + dst] = block_mv[(slice(None),) + src]
                coverage[(local_t,) + dst] += 1
    if not np.all(coverage == 1):
        raise AssertionError("Inference tiles did not cover ROI exactly once")

    candidates = []
    window = tuple(config.model.nms_window_shape_test or
                   config.model.nms_window_shape)
    for local_t in range(size[0]):
        volume = ci[local_t]
        maxima = (volume >= score_threshold) & (
            volume == maximum_filter(volume, size=window, mode="constant",
                                     cval=-np.inf))
        components, count = label(maxima)
        points = []
        for component in range(1, count + 1):
            coords = np.argwhere(components == component)
            local = coords[np.argmax(volume[tuple(coords.T)])]
            points.append((float(volume[tuple(local)]), local))
        points.sort(key=lambda item: item[0], reverse=True)
        for score, local in points[:max_candidates_per_frame]:
            zyx = start[1:] + local
            vector = mv[(slice(None), local_t) + tuple(local)] / 32.0
            candidates.append({"node_id": len(candidates),
                               "t": int(start[0] + local_t),
                               "z": int(zyx[0]), "y": int(zyx[1]),
                               "x": int(zyx[2]), "score": score,
                               "mv_z_um": float(vector[0]),
                               "mv_y_um": float(vector[1]),
                               "mv_x_um": float(vector[2])})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "candidate_nodes.csv", NODE_FIELDS, candidates)
    np.savez_compressed(output_dir / "prediction_maps.npz", cell_indicator=ci,
                        movement_vectors=mv)
    return candidates, {"tile_output_shape": output_shape.tolist(),
                        "tile_count": int(np.prod(np.ceil(size[1:] / step)) * size[0]),
                        "candidate_count": len(candidates)}


def candidate_edges(nodes, max_parent_distance_um=12.0, max_parents=3):
    by_t = {}
    for node in nodes:
        by_t.setdefault(int(node["t"]), []).append(node)
    result = []
    for t, children in by_t.items():
        parents = by_t.get(t - 1, [])
        if not parents:
            continue
        positions = np.array([[p[k] for k in ("z", "y", "x")]
                              for p in parents], dtype=float) * SPACING_UM
        tree = cKDTree(positions)
        for child in children:
            location = np.array([child[k] for k in ("z", "y", "x")],
                                dtype=float) * SPACING_UM
            movement = np.array([child[k] for k in
                                 ("mv_z_um", "mv_y_um", "mv_x_um")], dtype=float)
            inferred_parent = location + movement
            nearby = tree.query_ball_point(inferred_parent, max_parent_distance_um)
            nearby.sort(key=lambda j: np.linalg.norm(positions[j] - inferred_parent))
            for j in nearby[:max_parents]:
                result.append({"source_id": int(parents[j]["node_id"]),
                               "target_id": int(child["node_id"]),
                               "prediction_distance_um": float(
                                   np.linalg.norm(positions[j] - inferred_parent))})
    return result


def solve_basic_ilp(nodes, edges, time_limit=300, selection_constant=12.0,
                    weight_node_score=17.0, track_cost=7.0,
                    division_cost=1.0, edge_weight=0.35,
                    first_frame=None):
    """Linajea basic binary-forest constraints with configurable costs."""
    n, m = len(nodes), len(edges)
    if not n:
        return [], [], {"solver_status": "no_candidates"}
    ids = {int(node["node_id"]): i for i, node in enumerate(nodes)}
    # Variables: selected node, appearance, division, selected edge.
    edge_base = 3 * n
    costs = np.zeros(edge_base + m, dtype=float)
    first_t = (min(int(node["t"]) for node in nodes)
               if first_frame is None else first_frame)
    for i, node in enumerate(nodes):
        costs[i] = selection_constant - weight_node_score * float(node["score"])
        costs[n + i] = 0.0 if int(node["t"]) == first_t else track_cost
        costs[2 * n + i] = division_cost
    for e, edge in enumerate(edges):
        costs[edge_base + e] = edge_weight * float(edge["prediction_distance_um"])
    incoming, outgoing = [[] for _ in nodes], [[] for _ in nodes]
    for e, edge in enumerate(edges):
        p, c = ids[int(edge["source_id"])], ids[int(edge["target_id"])]
        if int(nodes[c]["t"]) - int(nodes[p]["t"]) != 1:
            raise ValueError("Candidate edge must advance exactly one frame")
        outgoing[p].append(e)
        incoming[c].append(e)
    rows, cols, values, lower, upper = [], [], [], [], []

    def add(terms, lo=-np.inf, hi=np.inf):
        row = len(lower)
        for col, coefficient in terms:
            rows.append(row)
            cols.append(col)
            values.append(coefficient)
        lower.append(lo)
        upper.append(hi)

    for e, edge in enumerate(edges):
        p, c = ids[int(edge["source_id"])], ids[int(edge["target_id"])]
        add([(edge_base + e, 2), (p, -1), (c, -1)], hi=0)
    for i in range(n):
        add([(edge_base + e, 1) for e in incoming[i]] +
            [(n + i, 1), (i, -1)], lo=0, hi=0)
        next_edges = [(edge_base + e, 1) for e in outgoing[i]]
        add(next_edges + [(i, -2)], hi=0)
        add(next_edges + [(2 * n + i, -1)], hi=1)
        add(next_edges + [(2 * n + i, -2)], lo=0)
    matrix = coo_matrix((values, (rows, cols)),
                        shape=(len(lower), len(costs))).tocsr()
    outcome = milp(costs, integrality=np.ones(len(costs)),
                   bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
                   constraints=LinearConstraint(matrix, lower, upper),
                   options={"time_limit": time_limit, "mip_rel_gap": 0.05})
    if outcome.x is None:
        raise RuntimeError(f"ILP failed: {outcome.message}")
    selected_nodes = [node for i, node in enumerate(nodes) if outcome.x[i] > 0.5]
    selected_ids = {int(node["node_id"]) for node in selected_nodes}
    selected_edges = [edge for e, edge in enumerate(edges)
                      if outcome.x[edge_base + e] > 0.5]
    parent_count, child_count = {}, {}
    for edge in selected_edges:
        p, c = int(edge["source_id"]), int(edge["target_id"])
        assert p in selected_ids and c in selected_ids
        parent_count[c] = parent_count.get(c, 0) + 1
        child_count[p] = child_count.get(p, 0) + 1
    assert max(parent_count.values(), default=0) <= 1
    assert max(child_count.values(), default=0) <= 2
    gap = getattr(outcome, "mip_gap", None)
    return selected_nodes, selected_edges, {
        "solver_status": str(outcome.message),
        "solver_objective": float(outcome.fun),
        "solver_gap": float(gap) if gap is not None else None,
        "selected_nodes": len(selected_nodes), "selected_edges": len(selected_edges)}


def reconstruct(image_path, config_path, checkpoint_path, manifest_path,
                output_dir, score_threshold=0.2, max_candidates_per_frame=200,
                device=None):
    output_dir = Path(output_dir)
    nodes, diagnostics = predict_roi(
        image_path, config_path, checkpoint_path, manifest_path, output_dir,
        score_threshold, max_candidates_per_frame, device)
    edges = candidate_edges(nodes)
    write_rows(output_dir / "candidate_edges.csv", EDGE_FIELDS, edges)
    selected_nodes, selected_edges, status = solve_basic_ilp(nodes, edges)
    write_rows(output_dir / "pred_nodes.csv", NODE_FIELDS, selected_nodes)
    write_rows(output_dir / "pred_edges.csv", EDGE_FIELDS, selected_edges)
    diagnostics.update(status)
    diagnostics["fixed_parameters"] = {
        "nms_score_threshold": score_threshold,
        "max_candidates_per_frame": max_candidates_per_frame,
        "max_parent_distance_um": 12.0,
        "max_parents_per_candidate": 3,
        "node_cost": "12 - 17 * score",
        "appearance_cost": 7.0,
        "division_cost": 1.0,
        "edge_cost": "0.35 * prediction_distance_um",
        "milp_time_limit_seconds": 300,
    }
    (output_dir / "reconstruction.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics

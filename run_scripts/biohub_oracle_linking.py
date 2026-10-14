"""Compare distance and learned-movement linking with oracle Biohub centers.

The supplied GEFF node positions are used as detections. GEFF edges are read
only for evaluation; no linking method receives them as input.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import read_rows, write_rows
from linajea.biohub_reconstruct import NODE_FIELDS


SPACING_UM = np.array([1.625, 0.40625, 0.40625])
ORACLE_FIELDS = NODE_FIELDS
LINK_FIELDS = ("source_id", "target_id", "distance_um", "movement_error_um")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oracle_nodes(records, voxel_size):
    """Keep exact GT IDs and voxel centers, without retaining parent IDs."""
    spacing = np.asarray(voxel_size, dtype=float)
    nodes = []
    for record in records:
        world = np.array([record[k] for k in ("t", "z", "y", "x")])
        voxel = np.rint(world / spacing).astype(int)
        if not np.allclose(world, voxel * spacing, atol=1e-6, rtol=0):
            raise ValueError("Oracle GEFF center does not lie on image voxel grid")
        nodes.append({"node_id": int(record["cell_id"]),
                      **dict(zip(("t", "z", "y", "x"), map(int, voxel))),
                      "score": 0.0, "mv_z_um": 0.0,
                      "mv_y_um": 0.0, "mv_x_um": 0.0})
    return nodes


def sample_model_at_centers(image_path, config_path, checkpoint_path, nodes,
                            device="cuda"):
    """Use the same tile grid and normalization as full-sequence inference."""
    import torch
    from linajea.config import TrackingConfig
    from linajea.training.torch_model import UnetModelWrapper

    image = BiohubImage(image_path)
    config = TrackingConfig.from_file(str(config_path))
    model = UnetModelWrapper(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    input_shape = np.asarray(config.model.predict_input_shape, dtype=int)
    with torch.no_grad():
        trial = model(torch.zeros(tuple(input_shape), device=device))
    output_shape = np.asarray(trial[0].shape, dtype=int)
    step = output_shape[1:]
    if output_shape[0] != 1 or np.any(step <= 0) or input_shape[0] % 2 != 1:
        raise ValueError("Unsupported Linajea prediction shape")
    if np.any((input_shape[1:] - step) % 2):
        raise ValueError("Model input/output margins must be symmetric")
    halo = (input_shape[1:] - step) // 2
    t_halo = (input_shape[0] - 1) // 2
    spatial_shape = np.asarray(image.shape[1:], dtype=int)
    qlo, qhi = image.attrs["stats"]["q001"], image.attrs["stats"]["q099"]
    if qhi <= qlo:
        raise ValueError("Invalid Biohub normalization metadata")

    grouped = defaultdict(list)
    for node in nodes:
        point = np.array([node[a] for a in ("z", "y", "x")], dtype=int)
        if not 0 <= int(node["t"]) < image.shape[0] or np.any(point < 0) or np.any(point >= spatial_shape):
            raise ValueError("Oracle node outside image")
        origin = tuple((point // step * step).tolist())
        grouped[(int(node["t"]), origin)].append((node, point))

    with torch.no_grad():
        for index, ((t, origin), group) in enumerate(grouped.items(), start=1):
            out_global = np.asarray(origin, dtype=int)
            read_begin = out_global - halo
            read_end = read_begin + input_shape[1:]
            valid_begin = np.maximum(read_begin, 0)
            valid_end = np.minimum(read_end, spatial_shape)
            dest_begin = valid_begin - read_begin
            dest_end = dest_begin + valid_end - valid_begin
            dst = tuple(slice(int(a), int(b)) for a, b in zip(dest_begin, dest_end))
            src = tuple(slice(int(a), int(b)) for a, b in zip(valid_begin, valid_end))
            raw = np.zeros(tuple(input_shape), dtype=np.float32)
            for dt in range(input_shape[0]):
                ft = t + dt - t_halo
                if 0 <= ft < image.shape[0]:
                    values = image.read((slice(ft, ft + 1),) + src)[0].astype(np.float32)
                    raw[(dt,) + dst] = (np.clip(values, qlo / 2, qhi * 2) - qlo) / (qhi - qlo)
            result = model(torch.from_numpy(raw).to(device))
            ci = result[0][0].detach().cpu().numpy()
            movement = result[3][:, 0].detach().cpu().numpy()
            for node, point in group:
                local = tuple((point - out_global).tolist())
                node["score"] = float(ci[local])
                vector = movement[(slice(None),) + local] / 32.0
                for axis, value in zip(("mv_z_um", "mv_y_um", "mv_x_um"), vector):
                    node[axis] = float(value)
            if index % 100 == 0 or index == len(grouped):
                print(f"GT-center tiles: {index}/{len(grouped)}", flush=True)
    return {"tiles": len(grouped), "nodes": len(nodes),
            "output_step": step.tolist(), "normalization": [float(qlo), float(qhi)]}


def common_candidate_edges(nodes, max_distance_um=12.0):
    """Same physical-distance gate for every method; never consult GT edges."""
    by_t = defaultdict(list)
    for node in nodes:
        by_t[int(node["t"])].append(node)
    result = []
    for t, children in by_t.items():
        parents = by_t.get(t - 1, [])
        if not parents:
            continue
        parent_xyz = np.array([[p[a] for a in ("z", "y", "x")]
                               for p in parents], dtype=float) * SPACING_UM
        tree = cKDTree(parent_xyz)
        for child in children:
            point = np.array([child[a] for a in ("z", "y", "x")],
                             dtype=float) * SPACING_UM
            moved = point + np.array([child[a] for a in
                                      ("mv_z_um", "mv_y_um", "mv_x_um")])
            for j in tree.query_ball_point(point, max_distance_um):
                result.append({"source_id": int(parents[j]["node_id"]),
                               "target_id": int(child["node_id"]),
                               "distance_um": float(np.linalg.norm(parent_xyz[j] - point)),
                               "movement_error_um": float(np.linalg.norm(parent_xyz[j] - moved))})
    return result


def greedy_link(nodes, edges, distance_key, appearance_cost=7.0,
                division_cost=1.0, edge_weight=0.35):
    """Link children by best savings; at most one parent and two daughters."""
    first_t = min(int(n["t"]) for n in nodes)
    by_child = defaultdict(list)
    for edge in edges:
        by_child[int(edge["target_id"])].append(edge)
    ranking = []
    for node in nodes:
        child = int(node["node_id"])
        if int(node["t"]) == first_t:
            continue
        options = sorted(by_child[child], key=lambda e: (float(e[distance_key]),
                         int(e["source_id"])))
        if options:
            ranking.append((appearance_cost - edge_weight * float(options[0][distance_key]),
                            child, options))
    ranking.sort(key=lambda item: (-item[0], item[1]))
    daughter_count = defaultdict(int)
    chosen = []
    for _, _, options in ranking:
        best = None
        for edge in options:
            parent = int(edge["source_id"])
            if daughter_count[parent] >= 2:
                continue
            cost = edge_weight * float(edge[distance_key]) + (
                division_cost if daughter_count[parent] == 1 else 0.0)
            if best is None or (cost, parent) < (best[0], int(best[1]["source_id"])):
                best = (cost, edge)
        if best is not None and best[0] < appearance_cost:
            edge = best[1]
            chosen.append(edge)
            daughter_count[int(edge["source_id"])] += 1
    return chosen, {"solver_status": "greedy"}


def ilp_link(nodes, edges, distance_key, appearance_cost=7.0,
             division_cost=1.0, edge_weight=0.35, time_limit=300):
    """Linajea's basic forest constraints with every oracle node fixed on."""
    n, m = len(nodes), len(edges)
    if not n:
        return [], {"solver_status": "empty"}
    ids = {int(node["node_id"]): i for i, node in enumerate(nodes)}
    incoming, outgoing = [[] for _ in nodes], [[] for _ in nodes]
    costs = np.zeros(2 * n + m, dtype=float)
    first_t = min(int(node["t"]) for node in nodes)
    for i, node in enumerate(nodes):
        costs[i] = 0.0 if int(node["t"]) == first_t else appearance_cost
        costs[n + i] = division_cost
    for e, edge in enumerate(edges):
        p, c = ids[int(edge["source_id"])], ids[int(edge["target_id"])]
        if int(nodes[c]["t"]) - int(nodes[p]["t"]) != 1:
            raise ValueError("Oracle edge must advance one frame")
        outgoing[p].append(e)
        incoming[c].append(e)
        costs[2 * n + e] = edge_weight * float(edge[distance_key])
    row_ids, columns, values, lower, upper = [], [], [], [], []

    def add(terms, lo=-np.inf, hi=np.inf):
        row = len(lower)
        for col, value in terms:
            row_ids.append(row)
            columns.append(col)
            values.append(value)
        lower.append(lo)
        upper.append(hi)

    for i in range(n):
        # Every GT node exists: exactly one incoming link or an appearance.
        add([(2 * n + e, 1) for e in incoming[i]] + [(i, 1)], lo=1, hi=1)
        out = [(2 * n + e, 1) for e in outgoing[i]]
        add(out, hi=2)
        # Division variable is one exactly when a node has two daughters.
        add(out + [(n + i, -1)], hi=1)
        add(out + [(n + i, -2)], lo=0)
    matrix = coo_matrix((values, (row_ids, columns)),
                        shape=(len(lower), len(costs))).tocsr()
    outcome = milp(costs, integrality=np.ones(len(costs)),
                   bounds=Bounds(np.zeros(len(costs)), np.ones(len(costs))),
                   constraints=LinearConstraint(matrix, lower, upper),
                   options={"time_limit": time_limit, "mip_rel_gap": 0.01})
    if outcome.x is None:
        raise RuntimeError(f"Oracle edge ILP failed: {outcome.message}")
    selected = [edge for e, edge in enumerate(edges) if outcome.x[2 * n + e] > 0.5]
    gap = getattr(outcome, "mip_gap", None)
    return selected, {"solver_status": str(outcome.message),
                      "objective": float(outcome.fun),
                      "mip_gap": float(gap) if gap is not None else None}


def edge_metrics(chosen, gt_edge_set, oracle_node_count, candidate_edge_set):
    predicted = {(int(e["source_id"]), int(e["target_id"])) for e in chosen}
    recovered = len(predicted & gt_edge_set)
    return {"oracle_nodes": oracle_node_count, "oracle_node_recall": 1.0,
            "candidate_gt_edges": len(candidate_edge_set & gt_edge_set),
            "known_gt_edges": len(gt_edge_set),
            "selected_edges": len(predicted),
            "recovered_gt_edges": recovered,
            "known_edge_recall": recovered / len(gt_edge_set) if gt_edge_set else None,
            "confirmed_edge_fraction_lower_bound": recovered / len(predicted)
            if predicted else None,
            "unconfirmed_edges": len(predicted) - recovered}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max-distance-um", type=float, default=12.0)
    parser.add_argument("--appearance-cost", type=float, default=7.0)
    parser.add_argument("--division-cost", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=0.35)
    args = parser.parse_args()
    if args.max_distance_um <= 0:
        parser.error("--max-distance-um must be positive")
    if args.appearance_cost < 0 or args.division_cost < 0 or args.edge_weight < 0:
        parser.error("Linking costs must be nonnegative")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    if len(split["validation"]) != 2 or {s.split("_")[0] for s in split["validation"]} != {"6bba"}:
        raise ValueError("Expected two held-out 6bba validation samples")
    if {s.split("_")[0] for s in split["train"]} != {"44b6"}:
        raise ValueError("Expected training samples from 44b6 only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_signature = {"checkpoint_sha256": file_sha256(args.checkpoint),
                       "config_sha256": file_sha256(args.config)}
    summary = []
    for name in split["validation"]:
        sample_dir = args.output_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        image_path = args.data_dir / f"{name}.zarr"
        image = BiohubImage(image_path)
        if tuple(image.voxel_size) != (1, 52, 13, 13):
            raise ValueError(f"Unexpected Biohub voxel scale: {image.voxel_size}")
        gt, _ = read_geff_tracks(args.data_dir / f"{name}.geff",
                                 image.voxel_size, image.shape)
        gt_ids = {int(row["cell_id"]) for row in gt}
        gt_edges = {(int(row["parent_id"]), int(row["cell_id"]))
                    for row in gt if int(row["parent_id"]) in gt_ids}
        cache = sample_dir / "oracle_nodes_with_model.csv"
        signature_path = sample_dir / "oracle_cache_signature.json"
        position_signature = hashlib.sha256(json.dumps(
            [(int(row["cell_id"]), *(float(row[a]) for a in
              ("t", "z", "y", "x"))) for row in gt],
            sort_keys=True).encode()).hexdigest()
        expected_signature = {**model_signature,
                              "gt_positions_sha256": position_signature}
        if cache.exists():
            if not signature_path.exists() or json.loads(
                    signature_path.read_text(encoding="utf-8")) != expected_signature:
                raise ValueError(f"Stale oracle-node cache: {cache}")
            nodes = read_rows(cache)
            if {int(node["node_id"]) for node in nodes} != gt_ids:
                raise ValueError(f"Stale oracle-node cache: {cache}")
            sampling = {"cached": True, "nodes": len(nodes)}
        else:
            nodes = oracle_nodes(gt, image.voxel_size)
            sampling = sample_model_at_centers(image_path, args.config,
                                               args.checkpoint, nodes, args.device)
            write_rows(cache, ORACLE_FIELDS, nodes)
            signature_path.write_text(json.dumps(expected_signature, indent=2),
                                      encoding="utf-8")
        edges = common_candidate_edges(nodes, args.max_distance_um)
        write_rows(sample_dir / "common_candidate_edges.csv", LINK_FIELDS, edges)
        edge_set = {(int(e["source_id"]), int(e["target_id"])) for e in edges}
        for method, key, solver in (
                ("distance_greedy", "distance_um", greedy_link),
                ("movement_greedy", "movement_error_um", greedy_link),
                ("distance_ilp", "distance_um", ilp_link),
                ("movement_ilp", "movement_error_um", ilp_link)):
            chosen, status = solver(nodes, edges, key,
                                    args.appearance_cost, args.division_cost,
                                    args.edge_weight)
            write_rows(sample_dir / f"{method}_edges.csv", LINK_FIELDS, chosen)
            summary.append({"sample": name, "method": method,
                            "candidate_edges": len(edges),
                            **edge_metrics(chosen, gt_edges, len(nodes), edge_set),
                            **status})
        print(f"Finished {name}: {len(nodes)} oracle centers, "
              f"{len(gt_edges)} known GT edges", flush=True)
    path = args.output_dir / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(
            key for row in summary for key in row)))
        writer.writeheader()
        writer.writerows(summary)
    (args.output_dir / "settings.json").write_text(json.dumps({
        "max_distance_um": args.max_distance_um,
        "appearance_cost": args.appearance_cost,
        "division_cost": args.division_cost,
        "edge_weight": args.edge_weight,
        "checkpoint": str(args.checkpoint),
        "split": str(args.split),
        "sampling": "raw model output at exact GT centers",
        "evaluation_only_gt_edges": True}, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()

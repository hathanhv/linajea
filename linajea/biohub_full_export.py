"""Stream a Biohub sequence through Linajea and export the selected GEFF graph."""

import copy
import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import find_objects, label, maximum_filter

from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import write_rows
from linajea.biohub_reconstruct import (EDGE_FIELDS, NODE_FIELDS,
                                      candidate_edges, solve_basic_ilp)


def _frame_candidates(volume, movement, t, window, threshold, limit, first_id):
    maxima = (volume >= threshold) & (
        volume == maximum_filter(volume, size=window, mode="constant",
                                 cval=-np.inf))
    components, _ = label(maxima)
    points = []
    for index, bounds in enumerate(find_objects(components), start=1):
        if bounds is None:
            continue
        mask = components[bounds] == index
        scores = np.where(mask, volume[bounds], -np.inf)
        offset = np.asarray(np.unravel_index(np.argmax(scores), scores.shape))
        point = offset + np.asarray([axis.start for axis in bounds])
        points.append((float(volume[tuple(point)]), point))
    points.sort(key=lambda item: item[0], reverse=True)
    rows = []
    for score, point in points[:limit]:
        vector = movement[(slice(None),) + tuple(point)] / 32.0
        rows.append({"node_id": first_id + len(rows), "t": t,
                     "z": int(point[0]), "y": int(point[1]),
                     "x": int(point[2]), "score": score,
                     "mv_z_um": float(vector[0]),
                     "mv_y_um": float(vector[1]),
                     "mv_x_um": float(vector[2])})
    return rows


def predict_full_sequence(image_path, config_path, checkpoint_path, output_dir,
                          score_threshold=0.2, max_candidates_per_frame=200,
                          device=None):
    """Keep only one full output frame in RAM; cache completed candidates."""
    import torch
    from linajea.config import TrackingConfig
    from linajea.training.torch_model import UnetModelWrapper

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = BiohubImage(image_path)
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
    if output_shape[0] != 1 or np.any(output_shape[1:] <= 0):
        raise ValueError(f"Unexpected model output shape {output_shape}")
    step = output_shape[1:]
    if np.any((input_shape[1:] - step) % 2) or input_shape[0] % 2 != 1:
        raise ValueError("Model input/output margins are not symmetric")
    halo = (input_shape[1:] - step) // 2
    t_halo = (input_shape[0] - 1) // 2
    qlo, qhi = image.attrs["stats"]["q001"], image.attrs["stats"]["q099"]
    if qhi <= qlo:
        raise ValueError("Invalid normalization quantiles")
    window = tuple(config.model.nms_window_shape_test or
                   config.model.nms_window_shape)

    @lru_cache(maxsize=8)
    def frame(t):
        return image.read((slice(t, t + 1), slice(None),
                           slice(None), slice(None)))[0]

    nodes = []
    spatial_shape = np.asarray(image.shape[1:], dtype=int)
    with torch.no_grad():
        for t in range(image.shape[0]):
            ci = np.zeros(tuple(spatial_shape), dtype=np.float32)
            mv = np.zeros((3,) + tuple(spatial_shape), dtype=np.float32)
            coverage = np.zeros(tuple(spatial_shape), dtype=np.uint8)
            for out_global in product(*(range(0, int(length), int(stride))
                                        for length, stride in zip(spatial_shape,
                                                                  step))):
                out_global = np.asarray(out_global, dtype=int)
                read_begin = out_global - halo
                read_end = read_begin + input_shape[1:]
                valid_begin = np.maximum(read_begin, 0)
                valid_end = np.minimum(read_end, spatial_shape)
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
                result = model(torch.from_numpy(raw).to(device))
                block_ci = result[0][0].detach().cpu().numpy()
                block_mv = result[3][:, 0].detach().cpu().numpy()
                lengths = np.minimum(step, spatial_shape - out_global)
                dst = tuple(slice(int(a), int(a + n))
                            for a, n in zip(out_global, lengths))
                src = tuple(slice(0, int(n)) for n in lengths)
                ci[dst] = block_ci[src]
                mv[(slice(None),) + dst] = block_mv[(slice(None),) + src]
                coverage[dst] += 1
            if not np.all(coverage == 1):
                raise AssertionError(f"Inference tiles did not cover frame {t} once")
            nodes.extend(_frame_candidates(ci, mv, t, window,
                                           score_threshold,
                                           max_candidates_per_frame,
                                           len(nodes)))
            if (t + 1) % 5 == 0 or t == 0:
                write_rows(output_dir / "candidate_nodes_partial.csv",
                           NODE_FIELDS, nodes)
                print(f"Full inference: frame {t + 1}/{image.shape[0]}, "
                      f"candidates {len(nodes)}", flush=True)
    write_rows(output_dir / "candidate_nodes.csv", NODE_FIELDS, nodes)
    return nodes


def solve_components(nodes, edges, selection_constant, track_cost,
                     time_limit_per_component=300):
    """Decompose exact ILP by candidate graph components to limit memory."""
    ids = {int(node["node_id"]): index for index, node in enumerate(nodes)}
    roots = list(range(len(nodes)))

    def root(index):
        while roots[index] != index:
            roots[index] = roots[roots[index]]
            index = roots[index]
        return index

    for edge in edges:
        a, b = ids[int(edge["source_id"])], ids[int(edge["target_id"])]
        roots[root(a)] = root(b)
    groups = {}
    for node in nodes:
        groups.setdefault(root(ids[int(node["node_id"])]), []).append(node)
    edge_groups = {key: [] for key in groups}
    for edge in edges:
        edge_groups[root(ids[int(edge["source_id"])])].append(edge)
    selected_nodes, selected_edges, statuses = [], [], []
    for index, (key, subset) in enumerate(groups.items()):
        chosen_nodes, chosen_edges, status = solve_basic_ilp(
            subset, edge_groups[key], time_limit=time_limit_per_component,
            selection_constant=selection_constant, track_cost=track_cost,
            first_frame=0)
        selected_nodes.extend(chosen_nodes)
        selected_edges.extend(chosen_edges)
        statuses.append(status)
        if (index + 1) % 100 == 0:
            print(f"Solved components: {index + 1}/{len(groups)}", flush=True)
    return selected_nodes, selected_edges, statuses


def write_predicted_geff(path, nodes, edges, original_geff, image_path):
    """Write a Zarr v2 GEFF for broad napari-geff compatibility."""
    import zarr

    image = BiohubImage(image_path)
    node_ids = [int(node["node_id"]) for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("Predicted node IDs must be unique")
    node_set = set(node_ids)
    parent_count, child_count = {}, {}
    positions = {int(node["node_id"]): tuple(int(node[a]) for a in
                                               ("t", "z", "y", "x"))
                 for node in nodes}
    for position in positions.values():
        if any(v < 0 or v >= bound for v, bound in zip(position, image.shape)):
            raise ValueError(f"Predicted node outside image: {position}")
    for edge in edges:
        parent, child = int(edge["source_id"]), int(edge["target_id"])
        if parent not in node_set or child not in node_set:
            raise ValueError("Predicted edge refers to missing node")
        if positions[child][0] - positions[parent][0] != 1:
            raise ValueError("Predicted edge must advance one frame")
        parent_count[child] = parent_count.get(child, 0) + 1
        child_count[parent] = child_count.get(parent, 0) + 1
    if max(parent_count.values(), default=0) > 1 or max(child_count.values(), default=0) > 2:
        raise ValueError("Predicted graph violates lineage degree constraints")

    original = zarr.open_group(str(original_geff), mode="r")
    metadata = copy.deepcopy(original.attrs["geff"])
    metadata.setdefault("extra", {})["estimated_number_of_nodes"] = len(nodes)
    metadata["extra"]["source"] = "Linajea predicted graph"
    if nodes:
        for axis in metadata["axes"]:
            values = [int(node[axis["name"]]) for node in nodes]
            axis["min"], axis["max"] = min(values), max(values)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing GEFF: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = zarr.open_group(str(path), mode="w", zarr_format=2)
    output.attrs["geff"] = metadata
    output.create_array("nodes/ids", data=np.asarray(node_ids, dtype=np.uint64))
    output.create_array("edges/ids", data=np.asarray(
        [(int(edge["source_id"]), int(edge["target_id"])) for edge in edges],
        dtype=np.uint64).reshape(-1, 2))
    for axis in ("t", "z", "y", "x"):
        output.create_array(f"nodes/props/{axis}/values", data=np.asarray(
            [int(node[axis]) for node in nodes], dtype=np.int64))
    records, read_edges = read_geff_tracks(path, image.voxel_size, image.shape)
    if len(records) != len(nodes) or read_edges != len(edges):
        raise AssertionError("Exported GEFF failed round-trip validation")
    return path


def export_full_graph(image_path, original_geff, config_path, checkpoint_path,
                      calibration_path, output_dir, device=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_file = output_dir / "candidate_nodes.csv"
    if nodes_file.is_file():
        from linajea.biohub_random_hide import read_rows
        nodes = read_rows(nodes_file)
        print(f"Reusing {len(nodes)} cached full-sequence candidates", flush=True)
    else:
        nodes = predict_full_sequence(image_path, config_path, checkpoint_path,
                                      output_dir, device=device)
    edges = candidate_edges(nodes)
    write_rows(output_dir / "candidate_edges.csv", EDGE_FIELDS, edges)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    chosen = calibration["chosen"]
    selected_nodes, selected_edges, statuses = solve_components(
        nodes, edges, chosen["selection_constant"], chosen["track_cost"])
    write_rows(output_dir / "pred_nodes.csv", NODE_FIELDS, selected_nodes)
    write_rows(output_dir / "pred_edges.csv", EDGE_FIELDS, selected_edges)
    geff = write_predicted_geff(output_dir / "predicted_full.geff",
                                selected_nodes, selected_edges,
                                original_geff, image_path)
    report = {"candidate_nodes": len(nodes), "candidate_edges": len(edges),
              "selected_nodes": len(selected_nodes),
              "selected_edges": len(selected_edges),
              "component_count": len(statuses),
              "selection_constant": chosen["selection_constant"],
              "track_cost": chosen["track_cost"],
              "checkpoint": str(checkpoint_path),
              "geff": str(geff)}
    (output_dir / "export.json").write_text(json.dumps(report, indent=2),
                                            encoding="utf-8")
    return report

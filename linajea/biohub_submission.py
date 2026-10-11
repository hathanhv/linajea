"""Build a Biohub submission CSV from Linajea predictions on test images."""

import csv
import hashlib
import json
from pathlib import Path

from linajea.biohub_full_export import predict_full_sequence, solve_components
from linajea.biohub_io import BiohubImage
from linajea.biohub_random_hide import read_rows, write_rows
from linajea.biohub_reconstruct import EDGE_FIELDS, NODE_FIELDS, candidate_edges


COLUMNS = ("id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
           "source_id", "target_id")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_graph(nodes, edges, shape):
    """Validate one predicted graph before writing any submission rows."""
    positions = {}
    for node in nodes:
        node_id = int(node["node_id"])
        if node_id in positions:
            raise ValueError(f"Duplicate predicted node ID: {node_id}")
        position = tuple(int(node[axis]) for axis in ("t", "z", "y", "x"))
        if any(value < 0 or value >= bound for value, bound in zip(position, shape)):
            raise ValueError(f"Predicted node {node_id} outside image: {position}")
        positions[node_id] = position
    if not positions:
        raise ValueError("No selected nodes: dataset would be absent from CSV")

    parents, daughters, seen_edges = {}, {}, set()
    for edge in edges:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        if source not in positions or target not in positions:
            raise ValueError(f"Edge {source}->{target} references an absent node")
        if (source, target) in seen_edges:
            raise ValueError(f"Duplicate edge {source}->{target}")
        seen_edges.add((source, target))
        if positions[target][0] - positions[source][0] != 1:
            raise ValueError(f"Edge {source}->{target} must advance one frame")
        parents[target] = parents.get(target, 0) + 1
        daughters[source] = daughters.get(source, 0) + 1
        if parents[target] > 1 or daughters[source] > 2:
            raise ValueError("Predicted graph violates lineage degree constraints")


def append_graph_rows(writer, dataset, nodes, edges, next_id):
    """Write validated rows using the official node/edge CSV schema."""
    for node in sorted(nodes, key=lambda row: int(row["node_id"])):
        writer.writerow((next_id, dataset, "node", int(node["node_id"]),
                         *(int(node[axis]) for axis in ("t", "z", "y", "x")),
                         -1, -1))
        next_id += 1
    for edge in sorted(edges, key=lambda row: (int(row["source_id"]),
                                               int(row["target_id"]))):
        writer.writerow((next_id, dataset, "edge", -1, -1, -1, -1, -1,
                         int(edge["source_id"]), int(edge["target_id"])))
        next_id += 1
    return next_id


def build_submission(data_dir, config_path, checkpoint_path, output_dir,
                     selection_constant=7.0, track_cost=7.0,
                     score_threshold=0.2, max_candidates_per_frame=200,
                     device=None, sample=None):
    """Infer every test Zarr and write a checked submission.csv.

    ``sample`` is only for a local format smoke test. The returned CSV is
    partial in that mode and must not be submitted to the competition.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    config_path, checkpoint_path = Path(config_path), Path(checkpoint_path)
    datasets = sorted(path for path in data_dir.glob("*.zarr") if path.is_dir())
    if sample is not None:
        datasets = [path for path in datasets if path.stem == sample]
    if not datasets:
        raise FileNotFoundError(f"No test .zarr datasets in {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    settings = {"config_sha256": _sha256(config_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "selection_constant": float(selection_constant),
                "track_cost": float(track_cost),
                "score_threshold": float(score_threshold),
                "max_candidates_per_frame": int(max_candidates_per_frame)}
    report = []
    csv_path = output_dir / ("submission.csv" if sample is None
                             else f"smoke_{sample}.csv")
    temp_path = csv_path.with_suffix(".csv.tmp")
    try:
        with temp_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            next_id = 0
            for image_path in datasets:
                name = image_path.stem
                sample_dir = output_dir / "graphs" / name
                sample_dir.mkdir(parents=True, exist_ok=True)
                settings_path = sample_dir / "settings.json"
                if settings_path.is_file():
                    previous = json.loads(settings_path.read_text())
                    if previous != settings:
                        raise ValueError(f"Cached graph has different settings: {name}")
                else:
                    if (sample_dir / "candidate_nodes.csv").exists():
                        raise ValueError(f"Unverified candidate cache: {sample_dir}")
                    settings_path.write_text(json.dumps(settings, indent=2))

                selected_path = sample_dir / "pred_nodes.csv"
                selected_edges_path = sample_dir / "pred_edges.csv"
                if selected_path.is_file() and selected_edges_path.is_file():
                    selected = read_rows(selected_path)
                    selected_edges = read_rows(selected_edges_path)
                else:
                    candidate_path = sample_dir / "candidate_nodes.csv"
                    candidates = (read_rows(candidate_path) if candidate_path.is_file()
                                  else predict_full_sequence(
                                      image_path, config_path, checkpoint_path,
                                      sample_dir, score_threshold,
                                      max_candidates_per_frame, device))
                    edges = candidate_edges(candidates)
                    selected, selected_edges, _ = solve_components(
                        candidates, edges, selection_constant, track_cost)
                    write_rows(selected_path, NODE_FIELDS, selected)
                    write_rows(selected_edges_path, EDGE_FIELDS, selected_edges)

                shape = BiohubImage(image_path).shape
                validate_graph(selected, selected_edges, shape)
                next_id = append_graph_rows(writer, name, selected,
                                            selected_edges, next_id)
                report.append({"dataset": name, "nodes": len(selected),
                               "edges": len(selected_edges), "shape": shape})
                print(f"{name}: {len(selected)} nodes, "
                      f"{len(selected_edges)} edges", flush=True)
        temp_path.replace(csv_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    (output_dir / "submission_report.json").write_text(
        json.dumps({"settings": settings, "datasets": report,
                    "partial_smoke_test": sample is not None}, indent=2))
    return csv_path, report

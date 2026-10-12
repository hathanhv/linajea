"""Evaluate saved Biohub candidate and selected CSV graphs without inference."""

import argparse
import csv
import json
from pathlib import Path

from linajea.biohub_holdout import sparse_diagnostics
from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import read_rows
from linajea.biohub_reconstruct import candidate_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True,
                        help="Saved validation folder with SAMPLE/*.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    samples = split["validation"]
    if len(samples) != 2 or set(samples) & set(split["train"]):
        raise ValueError("Split must contain two held-out samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in samples:
        source = args.prediction_dir / name
        candidates = read_rows(source / "candidate_nodes.csv")
        edge_file = source / "candidate_edges.csv"
        candidate_graph_edges = (read_rows(edge_file) if edge_file.is_file()
                                 else candidate_edges(candidates))
        selected = read_rows(source / "pred_nodes.csv")
        selected_edges = read_rows(source / "pred_edges.csv")
        image = BiohubImage(args.data_dir / f"{name}.zarr")
        gt, _ = read_geff_tracks(args.data_dir / f"{name}.geff",
                                 image.voxel_size, image.shape)
        candidate_result = {str(int(radius)): sparse_diagnostics(
            candidates, candidate_graph_edges, gt, radius)
                            for radius in (2.0, 4.0, 7.0)}
        selected_result = {str(int(radius)): sparse_diagnostics(
            selected, selected_edges, gt, radius)
                           for radius in (2.0, 4.0, 7.0)}
        sample_out = args.output_dir / name
        sample_out.mkdir(parents=True, exist_ok=True)
        (sample_out / "candidate_sparse_diagnostics.json").write_text(
            json.dumps(candidate_result, indent=2), encoding="utf-8")
        (sample_out / "sparse_diagnostics.json").write_text(
            json.dumps(selected_result, indent=2), encoding="utf-8")
        c, s = candidate_result["7"], selected_result["7"]
        rows.append({
            "sample": name,
            "candidate_nodes": len(candidates),
            "candidate_edges": len(candidate_graph_edges),
            "candidate_matched_gt_nodes": c["matched_gt_nodes"],
            "candidate_node_recall": c["known_node_recall"],
            "candidate_matched_gt_edges": c["matched_gt_edges"],
            "candidate_edge_recall": c["known_edge_recall"],
            **{key: value for key, value in s.items()
               if key != "precision_note"},
        })
    destination = args.output_dir / "summary_7um.csv"
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(destination, flush=True)


if __name__ == "__main__":
    main()

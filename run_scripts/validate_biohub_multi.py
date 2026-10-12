"""Predict two held-out Biohub samples and report sparse-GT recovery."""

import argparse
import csv
import json
from pathlib import Path

from linajea.biohub_full_export import (predict_full_sequence,
                                        solve_components, write_predicted_geff)
from linajea.biohub_holdout import sparse_diagnostics
from linajea.biohub_io import BiohubImage, read_geff_tracks
from linajea.biohub_random_hide import read_rows, write_rows
from linajea.biohub_reconstruct import EDGE_FIELDS, NODE_FIELDS, candidate_edges
from linajea.biohub_submission import _sha256, validate_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-constant", type=float, default=7.0)
    parser.add_argument("--track-cost", type=float, default=7.0)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-candidates-per-frame", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    train, validation = set(split["train"]), split["validation"]
    if len(validation) != 2 or train & set(validation):
        raise ValueError("Expected two validation samples absent from training")
    output_dir = args.output_dir.resolve()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    settings = {"checkpoint_sha256": _sha256(args.checkpoint),
                "config_sha256": _sha256(args.config),
                "selection_constant": args.selection_constant,
                "track_cost": args.track_cost,
                "score_threshold": args.score_threshold,
                "max_candidates_per_frame": args.max_candidates_per_frame}
    rows = []
    for name in validation:
        image_path = args.data_dir / f"{name}.zarr"
        gt_path = args.data_dir / f"{name}.geff"
        image = BiohubImage(image_path)
        sample_dir = output_dir / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        settings_path = sample_dir / "settings.json"
        if settings_path.exists():
            if json.loads(settings_path.read_text()) != settings:
                raise ValueError(f"Cached prediction settings differ for {name}")
        else:
            if (sample_dir / "candidate_nodes.csv").exists():
                raise ValueError(f"Unverified candidate cache for {name}")
            settings_path.write_text(json.dumps(settings, indent=2))

        selected_path = sample_dir / "pred_nodes.csv"
        selected_edges_path = sample_dir / "pred_edges.csv"
        if selected_path.exists() and selected_edges_path.exists():
            selected = read_rows(selected_path)
            selected_edges = read_rows(selected_edges_path)
        else:
            candidate_path = sample_dir / "candidate_nodes.csv"
            nodes = (read_rows(candidate_path) if candidate_path.exists()
                     else predict_full_sequence(
                         image_path, args.config, args.checkpoint, sample_dir,
                         args.score_threshold, args.max_candidates_per_frame,
                         args.device))
            edges = candidate_edges(nodes)
            write_rows(sample_dir / "candidate_edges.csv", EDGE_FIELDS, edges)
            selected, selected_edges, statuses = solve_components(
                nodes, edges, args.selection_constant, args.track_cost)
            (sample_dir / "solver_status.json").write_text(
                json.dumps(statuses, indent=2))
            write_rows(selected_path, NODE_FIELDS, selected)
            write_rows(selected_edges_path, EDGE_FIELDS, selected_edges)
        validate_graph(selected, selected_edges, image.shape)
        geff_out = pred_dir / f"{name}.geff"
        if not geff_out.exists():
            # Ground truth is opened here only to copy GEFF metadata after
            # prediction and graph selection are complete.
            write_predicted_geff(geff_out, selected, selected_edges,
                                 gt_path, image_path)
        candidate_nodes_path = sample_dir / "candidate_nodes.csv"
        if not candidate_nodes_path.is_file():
            raise FileNotFoundError(
                f"Candidate graph is required for pre-ILP recall: "
                f"{candidate_nodes_path}")
        candidate_nodes = read_rows(candidate_nodes_path)
        candidate_edges_path = sample_dir / "candidate_edges.csv"
        if candidate_edges_path.is_file():
            candidate_graph_edges = read_rows(candidate_edges_path)
        else:
            candidate_graph_edges = candidate_edges(candidate_nodes)
            write_rows(candidate_edges_path, EDGE_FIELDS, candidate_graph_edges)
        gt_records, _ = read_geff_tracks(gt_path, image.voxel_size, image.shape)
        metrics = {str(int(radius)): sparse_diagnostics(
            selected, selected_edges, gt_records, radius)
                   for radius in (2.0, 4.0, 7.0)}
        candidate_metrics = {str(int(radius)): sparse_diagnostics(
            candidate_nodes, candidate_graph_edges, gt_records, radius)
                             for radius in (2.0, 4.0, 7.0)}
        (sample_dir / "sparse_diagnostics.json").write_text(
            json.dumps(metrics, indent=2))
        (sample_dir / "candidate_sparse_diagnostics.json").write_text(
            json.dumps(candidate_metrics, indent=2))
        at7 = metrics["7"]
        candidate_at7 = candidate_metrics["7"]
        rows.append({
            "sample": name,
            "candidate_nodes": len(candidate_nodes),
            "candidate_edges": len(candidate_graph_edges),
            "candidate_matched_gt_nodes": candidate_at7["matched_gt_nodes"],
            "candidate_node_recall": candidate_at7["known_node_recall"],
            "candidate_matched_gt_edges": candidate_at7["matched_gt_edges"],
            "candidate_edge_recall": candidate_at7["known_edge_recall"],
            **{key: value for key, value in at7.items()
               if key != "precision_note"},
        })
        print(name, "7 um candidate cell",
              candidate_at7["matched_gt_nodes"], "/",
              candidate_at7["known_gt_nodes"], "candidate edge",
              candidate_at7["matched_gt_edges"], "/",
              candidate_at7["known_gt_edges"], "selected cell",
              at7["matched_gt_nodes"], "/",
              at7["known_gt_nodes"], "edge", at7["matched_gt_edges"], "/",
              at7["known_gt_edges"], flush=True)
    with (output_dir / "summary_7um.csv").open("w", newline="",
                                                 encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Predicted GEFF directory:", pred_dir)
    print("Diagnostic summary:", output_dir / "summary_7um.csv")


if __name__ == "__main__":
    main()

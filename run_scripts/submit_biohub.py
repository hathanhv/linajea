"""Predict Biohub test images and create a Kaggle submission CSV."""

import argparse
from pathlib import Path

from linajea.biohub_submission import build_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing test .zarr images")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selection-constant", type=float, default=7.0)
    parser.add_argument("--track-cost", type=float, default=7.0)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--max-candidates-per-frame", type=int, default=200)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--sample", help="Smoke test one dataset; writes smoke_NAME.csv")
    args = parser.parse_args()
    csv_path, report = build_submission(
        args.data_dir, args.config, args.checkpoint, args.output_dir,
        args.selection_constant, args.track_cost, args.score_threshold,
        args.max_candidates_per_frame, args.device, args.sample)
    print(f"CSV: {csv_path}; datasets: {len(report)}", flush=True)


if __name__ == "__main__":
    main()

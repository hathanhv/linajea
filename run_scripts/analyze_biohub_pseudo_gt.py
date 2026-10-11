"""Analyze cached Biohub candidate graphs without training or inference."""

import argparse
from pathlib import Path

from linajea.biohub_pseudo_analysis import analyze_cached


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Mounted prior Kaggle notebook output root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path,
                        help="Optional Biohub train directory for held-out metrics")
    parser.add_argument("--selection-constant", type=float, default=7.0)
    parser.add_argument("--track-cost", type=float, default=7.0)
    parser.add_argument("--time-limit-per-component", type=int, default=300)
    args = parser.parse_args()
    result = analyze_cached(
        args.source_dir, args.output_dir, args.gt_dir,
        args.selection_constant, args.track_cost,
        args.time_limit_per_component)
    print(f"Saved {len(result)} summary rows to "
          f"{args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()

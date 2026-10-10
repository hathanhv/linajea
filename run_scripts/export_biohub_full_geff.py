"""Export a 1500-iteration Biohub checkpoint as a full-sequence GEFF graph."""

import argparse
import json
from pathlib import Path

from linajea.biohub_full_export import export_full_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample", default="6bba_05b6850b")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory containing train/ and prediction_calibrated/")
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    root = args.run_dir.resolve()
    report = export_full_graph(
        args.data_dir / f"{args.sample}.zarr",
        args.data_dir / f"{args.sample}.geff",
        root / "train" / "config.toml",
        root / "train" / f"train_net_checkpoint_{args.iterations}",
        root / "prediction_calibrated" / "calibration.json",
        root / "full_export", device=args.device)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

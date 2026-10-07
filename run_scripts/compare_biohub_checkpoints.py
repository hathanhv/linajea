"""Compare saved Biohub checkpoints without training again."""

import argparse
import json
from pathlib import Path

from linajea.biohub_checkpoint_compare import compare_checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", nargs="+", type=int,
                        default=(500, 1000, 1500, 2000))
    parser.add_argument("--reuse-iteration", type=int, default=2000)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    report, summaries = compare_checkpoints(
        args.data_dir, args.output_dir, args.iterations,
        args.reuse_iteration, args.device)
    print(json.dumps({"report": report, "summary_4um": summaries}, indent=2))


if __name__ == "__main__":
    main()

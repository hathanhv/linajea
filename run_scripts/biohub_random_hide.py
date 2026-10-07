"""Run the split, prediction, evaluation, and visualization pilot stages."""

import argparse
import json
from pathlib import Path

from linajea.biohub_random_hide import create_split, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("split", "reconstruct", "evaluate", "plot"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample", default="6bba_05b6850b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hide-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if args.stage == "split":
        if not 0 < args.hide_rate < 1:
            parser.error("--hide-rate must be between zero and one")
        result = create_split(args.data_dir, args.sample, root,
                              args.hide_rate, args.seed)
        print(json.dumps({k: result[k] for k in
                          ("sample", "actual_hide_rate", "roi_start",
                           "roi_shape", "roi_gt_count")}, indent=2))
    elif args.stage == "reconstruct":
        from linajea.biohub_reconstruct import reconstruct
        result = reconstruct(args.data_dir / f"{args.sample}.zarr",
                             root / "train" / "config.toml",
                             root / "train" /
                             f"train_net_checkpoint_{args.iterations}",
                             root / "split.json", root / "prediction",
                             device=args.device)
        print(json.dumps(result, indent=2))
    elif args.stage == "evaluate":
        result = evaluate(args.data_dir, root / "split.json",
                          root / "prediction" / "pred_nodes.csv",
                          root / "prediction" / "pred_edges.csv",
                          root / "evaluation")
        print(json.dumps(result, indent=2))
    else:
        from linajea.biohub_visualize import plot_comparison
        path = plot_comparison(args.data_dir, root / "split.json",
                               root / "prediction" / "pred_nodes.csv",
                               root / "prediction" / "pred_edges.csv",
                               root / "plots")
        print("Comparison image:", path)


if __name__ == "__main__":
    main()

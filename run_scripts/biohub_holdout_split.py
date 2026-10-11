"""Create an eight-train/two-validation embryo-disjoint Biohub split."""

import argparse
from pathlib import Path

from linajea.biohub_holdout import choose_samples, save_split, validate_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-embryo", default="6bba")
    parser.add_argument("--val-embryo", default="44b6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split = choose_samples(args.data_dir, args.train_embryo,
                           args.val_embryo, seed=args.seed)
    details = validate_split(args.data_dir, split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_split(args.output, split, details)
    print("Train:", split["train"])
    print("Validation:", split["validation"])
    print("Manifest:", args.output)


if __name__ == "__main__":
    main()

"""Train Linajea on eight full-annotation Biohub training samples."""

import argparse
import json
import logging
import os
from pathlib import Path

import toml
import torch

from linajea.biohub_holdout import validate_split


TEMPLATE = (Path(__file__).resolve().parent.parent / "examples" /
            "biohub" / "config_train_smoke.toml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16000)
    parser.add_argument("--checkpoint-stride", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.checkpoint_stride < 1:
        parser.error("Iterations and checkpoint stride must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("No CUDA GPU; enable one or pass --allow-cpu")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    details = validate_split(args.data_dir, split)
    if len(split["train"]) != 8:
        parser.error("Expected eight training samples")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config_dict = toml.load(TEMPLATE)
    config_dict["general"]["seed"] = args.seed
    config_dict["train"]["max_iterations"] = args.iterations
    config_dict["train"]["checkpoint_stride"] = args.checkpoint_stride
    config_dict["train_data"]["data_sources"] = [
        {"tracksfile": str((args.data_dir / f"{name}.geff").resolve()),
         "datafile": {"filename": str((args.data_dir / f"{name}.zarr").resolve()),
                      "array": "0"}}
        for name in split["train"]]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.toml"
    if config_path.exists():
        old = toml.load(config_path)
        if old != config_dict:
            raise ValueError("Existing config differs; use a new output directory")
    else:
        with config_path.open("w", encoding="utf-8") as stream:
            toml.dump(config_dict, stream)

    from linajea.config import TrackingConfig
    from linajea.training import train

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.info("Training on %s; annotated nodes: %s", split["train"],
                 {name: details[name]["annotated_nodes"]
                  for name in split["train"]})
    os.chdir(output_dir)
    train(TrackingConfig.from_file(str(config_path)))
    checkpoint = output_dir / f"train_net_checkpoint_{args.iterations}"
    if not checkpoint.is_file():
        raise RuntimeError(f"No final checkpoint: {checkpoint}")
    print("Training complete:", checkpoint, flush=True)


if __name__ == "__main__":
    main()

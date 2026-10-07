"""Run a short Biohub training job and verify that a checkpoint was saved."""

import argparse
import logging
import os
from pathlib import Path

import toml
import torch

from linajea.biohub_io import BiohubImage, read_geff_tracks


TEMPLATE = (Path(__file__).resolve().parent.parent / "examples" /
            "biohub" / "config_train_smoke.toml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample", default="6bba_05b6850b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--tracksfile", type=Path,
                        help="Visible GEFF to use instead of the original annotations")
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Permit a slow CPU-only run for debugging")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.checkpoint_stride < 1:
        parser.error("--checkpoint-stride must be positive")
    if not torch.cuda.is_available() and not args.allow_cpu:
        parser.error("No CUDA GPU detected. Enable a Kaggle GPU or pass --allow-cpu")

    image_path = args.data_dir / f"{args.sample}.zarr"
    graph_path = args.tracksfile or args.data_dir / f"{args.sample}.geff"
    if not image_path.is_dir() or not graph_path.is_dir():
        parser.error(f"Expected paired stores: {image_path} and {graph_path}")
    image = BiohubImage(image_path)
    records, edges = read_geff_tracks(graph_path, image.voxel_size, image.shape)
    if not records or not edges:
        parser.error("The selected GEFF needs annotated nodes and edges")
    if not all(k in image.attrs["stats"] for k in ("q001", "q099")):
        parser.error("The image is missing 0.01/0.99 quantiles")

    config_dict = toml.load(TEMPLATE)
    config_dict["train"]["max_iterations"] = args.iterations
    config_dict["train"]["checkpoint_stride"] = args.checkpoint_stride
    source = config_dict["train_data"]["data_sources"][0]
    source["tracksfile"] = str(graph_path.resolve())
    source["datafile"]["filename"] = str(image_path.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.toml"
    with config_path.open("w", encoding="utf-8") as stream:
        toml.dump(config_dict, stream)

    from linajea.config import TrackingConfig
    from linajea.training import train

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.info("Train sample %s: %d nodes, %d edges, device=%s",
                 args.sample, len(records), edges,
                 "cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(output_dir)
    config = TrackingConfig.from_file(str(config_path))
    train(config)

    checkpoint = output_dir / f"train_net_checkpoint_{args.iterations}"
    if not checkpoint.is_file():
        raise RuntimeError(f"Training ended without checkpoint {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not all(torch.isfinite(value).all()
               for value in state["model_state_dict"].values()
               if torch.is_tensor(value)):
        raise FloatingPointError("Checkpoint contains non-finite model weights")
    print("Training smoke test passed; checkpoint:", checkpoint, flush=True)


if __name__ == "__main__":
    main()

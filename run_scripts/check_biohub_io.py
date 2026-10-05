"""Read one Biohub sample and request a small Linajea/Gunpowder batch.

Example on Kaggle:
python run_scripts/check_biohub_io.py --data-dir \
  /kaggle/input/competitions/biohub-cell-tracking-during-development/train
"""

import argparse
from pathlib import Path

import numpy as np

from linajea.biohub_io import BiohubImage, read_geff_tracks


def choose_sample(data_dir, sample):
    if sample:
        stem = sample.removesuffix(".zarr").removesuffix(".geff")
        image_path = data_dir / f"{stem}.zarr"
    else:
        pairs = [p for p in sorted(data_dir.glob("*.zarr"))
                 if p.with_suffix(".geff").is_dir()]
        if not pairs:
            raise FileNotFoundError(f"No paired .zarr/.geff in {data_dir}")
        image_path = pairs[0]
    geff_path = image_path.with_suffix(".geff")
    if not image_path.is_dir() or not geff_path.is_dir():
        raise FileNotFoundError(f"Missing pair: {image_path}, {geff_path}")
    return image_path, geff_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing paired .zarr and .geff stores")
    parser.add_argument("--sample", help="Sample stem; defaults to first pair")
    args = parser.parse_args()
    image_path, geff_path = choose_sample(args.data_dir, args.sample)

    image = BiohubImage(image_path)
    records, num_edges = read_geff_tracks(
        geff_path, image.voxel_size, image.shape)
    if not records:
        raise ValueError(f"No annotated nodes in {geff_path}")
    first_chunk = image.read(tuple(slice(0, min(s, n)) for s, n in
                                   zip(image.shape, (1, 4, 16, 16))))
    print("Image:", image_path)
    print("Shape, dtype, voxel size:", image.shape, image.array.dtype,
          image.voxel_size)
    print("Quantiles q001/q099:", image.attrs["stats"].get("q001"),
          image.attrs["stats"].get("q099"))
    print("Small image chunk:", first_chunk.shape, first_chunk.dtype,
          "range", (int(first_chunk.min()), int(first_chunk.max())))
    print("GEFF:", geff_path, "nodes", len(records), "edges", num_edges)

    # Import only after the format checks, so dependency failures remain clear.
    import gunpowder as gp
    from linajea.biohub_gp import BiohubTracksSource, image_source

    raw = gp.ArrayKey("RAW")
    tracks = gp.GraphKey("TRACKS")
    voxel_size = gp.Coordinate(image.voxel_size)
    anchor = records[len(records) // 2]
    voxel_position = (anchor["t"],
                      int(round(anchor["z"] / voxel_size[1])),
                      int(round(anchor["y"] / voxel_size[2])),
                      int(round(anchor["x"] / voxel_size[3])))
    patch_shape = tuple(min(s, p) for s, p in
                        zip(image.shape, (3, 8, 64, 64)))
    patch_begin = tuple(max(0, min(int(v) - p // 2, s - p))
                        for v, p, s in
                        zip(voxel_position, patch_shape, image.shape))
    roi = gp.Roi(gp.Coordinate(patch_begin) * voxel_size,
                 gp.Coordinate(patch_shape) * voxel_size)
    full_roi = gp.Roi(gp.Coordinate(image.roi_offset),
                      gp.Coordinate(image.roi_shape))

    source = (
        (image_source(str(image_path), {raw: "0"},
                      {raw: gp.ArraySpec(interpolatable=True,
                                         voxel_size=voxel_size)}),
         BiohubTracksSource(str(geff_path), tracks,
                            voxel_size=image.voxel_size,
                            points_spec=gp.GraphSpec(roi=full_roi))) +
        gp.MergeProvider()
    )
    request = gp.BatchRequest()
    request[raw] = gp.ArraySpec(roi=roi)
    request[tracks] = gp.GraphSpec(roi=roi)
    with gp.build(source):
        batch = source.request_batch(request)

    array = batch.arrays[raw]
    graph = batch.graphs[tracks]
    num_batch_nodes = graph.num_vertices()
    if tuple(array.data.shape) != patch_shape:
        raise AssertionError("Gunpowder array shape does not match requested ROI")
    if num_batch_nodes == 0:
        raise AssertionError("Gunpowder graph batch has no annotated nodes")
    if not np.issubdtype(array.data.dtype, np.number):
        raise AssertionError("Gunpowder array has nonnumeric dtype")
    first_node = next(graph.nodes)
    relative_voxel = tuple(int(round((coordinate - start) / spacing))
                           for coordinate, start, spacing in
                           zip(first_node.location, roi.get_begin(), voxel_size))
    if any(index < 0 or index >= extent
           for index, extent in zip(relative_voxel, array.data.shape)):
        raise AssertionError("GEFF centroid does not align with image batch")
    print("Gunpowder batch:", array.data.shape, array.data.dtype,
          "ROI", roi, "nodes", num_batch_nodes)
    print("First centroid at batch voxel:", relative_voxel)
    print("Biohub I/O smoke test passed")


if __name__ == "__main__":
    main()

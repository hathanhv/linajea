"""Read Biohub's Zarr v3 images and GEFF lineage annotations.

Spatial world coordinates use units of 1/32 micrometer. This keeps the
Biohub voxel spacing exact while satisfying Gunpowder's integer ROI grid.
"""

import json
from pathlib import Path

import numpy as np


SPATIAL_UNITS_PER_MICROMETER = 32


def is_biohub_zarr(filename):
    """Recognize Biohub image stores without importing the legacy Zarr stack."""
    metadata = Path(filename) / "zarr.json"
    if not metadata.is_file():
        return False
    with metadata.open(encoding="utf-8") as stream:
        root = json.load(stream)
    return (root.get("zarr_format") == 3 and
            root.get("node_type") == "group" and
            bool(root.get("attributes", {}).get("multiscales")))


class BiohubImage:
    """Metadata and lazy array access for one Biohub image store."""

    def __init__(self, filename, array="0"):
        if not is_biohub_zarr(filename):
            raise ValueError(f"Not a Biohub Zarr v3 image: {filename}")
        import zarr

        self.filename = str(filename)
        self.array_name = str(array)
        self.root = zarr.open_group(self.filename, mode="r")
        self.array = self.root[self.array_name]
        if self.array.ndim != 4:
            raise ValueError("Biohub image must have axes (T, Z, Y, X)")

        multiscales = self.root.attrs["multiscales"]
        dataset = next((d for d in multiscales[0]["datasets"]
                        if d["path"] == self.array_name), None)
        if dataset is None:
            raise ValueError(f"Array {self.array_name!r} has no scale metadata")
        axes = [axis["name"].upper() for axis in multiscales[0]["axes"]]
        if axes != ["T", "Z", "Y", "X"]:
            raise ValueError(f"Unsupported image axes: {axes}")

        scale = next((item["scale"] for item in dataset["coordinateTransformations"]
                      if item["type"] == "scale"), None)
        if scale is None or len(scale) != 4 or scale[0] != 1:
            raise ValueError("Expected a four-dimensional scale with one frame per T")
        spatial = np.asarray(scale[1:], dtype=np.float64)
        spatial_units = spatial * SPATIAL_UNITS_PER_MICROMETER
        if np.any(spatial <= 0) or not np.allclose(spatial_units,
                                                   np.rint(spatial_units),
                                                   rtol=0, atol=1e-8):
            raise ValueError("Spatial voxel scale is not representable in 1/32 um")
        self.voxel_size = (1,) + tuple(int(v) for v in np.rint(spatial_units))
        self.shape = tuple(int(v) for v in self.array.shape)
        self.roi_offset = (0, 0, 0, 0)
        self.roi_shape = tuple(s * v for s, v in zip(self.shape, self.voxel_size))

        quantiles = self.root.attrs.get("image_statistics", {}).get("quantiles", {})
        stats = {"dtype": np.dtype(self.array.dtype)}
        for source, target in (("0.01", "q001"), ("0.99", "q099")):
            if source in quantiles:
                stats[target] = float(quantiles[source])
        self.attrs = {"stats": stats}

    def read(self, slices):
        """Read only the requested voxel region."""
        return np.asarray(self.array[slices])


def read_geff_tracks(filename, voxel_size, image_shape=None):
    """Return GEFF nodes as Linajea track records in integer world units.

    GEFF node positions are voxel indices. Edges run from parent to child.
    Every connected lineage receives the root node ID as its track_id.
    """
    import zarr

    group = zarr.open_group(str(filename), mode="r")
    geff_meta = group.attrs.get("geff", {})
    if not geff_meta.get("directed", False):
        raise ValueError("Biohub GEFF graph must be directed")
    axes = geff_meta.get("axes", [])
    if [axis.get("name", "").lower() for axis in axes] != ["t", "z", "y", "x"]:
        raise ValueError("GEFF axes must be (t, z, y, x)")
    expected_scale = (1,) + tuple(v / SPATIAL_UNITS_PER_MICROMETER
                                  for v in voxel_size[1:])
    if not np.allclose([axis.get("scale", 1) for axis in axes],
                       expected_scale, rtol=0, atol=1e-8):
        raise ValueError("GEFF spatial scale differs from image voxel scale")
    ids = np.asarray(group["nodes/ids"])
    positions = np.stack([np.asarray(group[f"nodes/props/{axis}/values"])
                          for axis in ("t", "z", "y", "x")], axis=1)
    edges = np.asarray(group["edges/ids"])
    if positions.shape != (len(ids), 4) or edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("Malformed GEFF node or edge arrays")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("GEFF contains duplicate node IDs")
    if not np.all(np.isfinite(positions)) or np.any(positions < 0):
        raise ValueError("GEFF contains invalid node positions")
    if not np.allclose(positions[:, 0], np.rint(positions[:, 0])):
        raise ValueError("GEFF time positions must be integer frames")
    if image_shape is not None and np.any(positions >= np.asarray(image_shape)):
        raise ValueError("GEFF node lies outside the image")
    if len(voxel_size) != 4:
        raise ValueError("voxel_size must have (T, Z, Y, X) entries")

    index = {int(node_id): i for i, node_id in enumerate(ids)}
    parent = {}
    children = {}
    seen_edges = set()
    for source, target in edges:
        source, target = int(source), int(target)
        if source not in index or target not in index:
            raise ValueError(f"GEFF edge {source}->{target} refers to a missing node")
        if (source, target) in seen_edges:
            raise ValueError(f"Duplicate GEFF edge {source}->{target}")
        seen_edges.add((source, target))
        if positions[index[target], 0] - positions[index[source], 0] != 1:
            raise ValueError(f"GEFF edge {source}->{target} must advance one frame")
        if target in parent:
            raise ValueError(f"GEFF node {target} has more than one parent")
        parent[target] = source
        children[source] = children.get(source, 0) + 1
        if children[source] > 2:
            raise ValueError(f"GEFF node {source} has more than two children")

    root_for = {}
    records = []
    for i in np.argsort(positions[:, 0], kind="stable"):
        node_id = int(ids[i])
        parent_id = parent.get(node_id, -1)
        root_for[node_id] = root_for[parent_id] if parent_id != -1 else node_id
        world = positions[i] * np.asarray(voxel_size)
        records.append({"t": int(world[0]),
                        "z": float(world[1]),
                        "y": float(world[2]),
                        "x": float(world[3]),
                        "cell_id": node_id,
                        "parent_id": parent_id,
                        "track_id": root_for[node_id]})
    return records, len(edges)

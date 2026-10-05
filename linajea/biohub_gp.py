"""Gunpowder sources for Biohub Zarr v3 images and GEFF graphs."""

import numpy as np
import gunpowder as gp

from linajea.biohub_io import BiohubImage, is_biohub_zarr, read_geff_tracks


class BiohubZarrSource(gp.BatchProvider):
    """Serve requested image ROIs lazily from a Biohub Zarr v3 store."""

    def __init__(self, filename, datasets, array_specs=None):
        self.filename = filename
        self.datasets = datasets
        self.array_specs = array_specs or {}
        self.images = {}

    def setup(self):
        for key, array_name in self.datasets.items():
            image = BiohubImage(self.filename, array_name)
            self.images[key] = image
            spec = self.array_specs.get(key, gp.ArraySpec()).copy()
            if spec.voxel_size is not None and tuple(spec.voxel_size) != image.voxel_size:
                raise ValueError("Configured voxel size differs from Biohub metadata")
            spec.voxel_size = gp.Coordinate(image.voxel_size)
            spec.roi = gp.Roi(gp.Coordinate(image.roi_offset),
                              gp.Coordinate(image.roi_shape))
            spec.dtype = np.dtype(image.array.dtype)
            self.provides(key, spec)

    def provide(self, request):
        batch = gp.Batch()
        for key, image in self.images.items():
            if key not in request:
                continue
            requested = request[key].roi
            begin = tuple(requested.get_begin())
            shape = tuple(requested.get_shape())
            voxel_size = image.voxel_size
            if any(b < 0 or b % v != 0 or s % v != 0 or
                   b + s > extent for b, s, v, extent in
                   zip(begin, shape, voxel_size, image.roi_shape)):
                raise ValueError(f"Requested ROI {requested} is outside or off the Biohub grid")
            slices = tuple(slice(b // v, (b + s) // v)
                           for b, s, v in zip(begin, shape, voxel_size))
            data = image.read(slices)
            spec = self.spec[key].copy()
            spec.roi = requested.copy()
            batch.arrays[key] = gp.Array(data, spec)
        return batch


def image_source(filename, datasets, array_specs=None):
    """Use Biohub's reader for v3 images and the original reader otherwise."""
    if is_biohub_zarr(filename):
        return BiohubZarrSource(filename, datasets, array_specs)
    return gp.ZarrSource(filename, datasets=datasets, array_specs=array_specs)


class BiohubTracksSource(gp.BatchProvider):
    """Provide GEFF centroids with the attributes Linajea's loss expects."""

    def __init__(self, filename, points, voxel_size, points_spec,
                 use_radius=False, attr_filter=None):
        self.filename = filename
        self.points = points
        self.voxel_size = tuple(voxel_size)
        self.points_spec = points_spec
        self.use_radius = use_radius
        self.attr_filter = attr_filter or {}

    def setup(self):
        self.records, _ = read_geff_tracks(self.filename, self.voxel_size)
        child_counts = {}
        for record in self.records:
            parent_id = record["parent_id"]
            if parent_id != -1:
                child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
        self.child_counts = child_counts
        self.provides(self.points, self.points_spec)

    def provide(self, request):
        roi = request[self.points].roi
        nodes = []
        for record in self.records:
            parent_id = record["parent_id"]
            if self.child_counts.get(record["cell_id"], 0) == 2:
                div_state = 1
            elif parent_id != -1 and self.child_counts.get(parent_id, 0) == 2:
                div_state = 2
            else:
                div_state = 0
            if any(key != "div_state" or div_state != value
                   for key, value in self.attr_filter.items()):
                continue
            location = np.asarray([record[axis] for axis in ("t", "z", "y", "x")],
                                  dtype=np.float32)
            if not roi.contains(gp.Coordinate(location)):
                continue
            radius = None
            if isinstance(self.use_radius, dict):
                radius_by_time = {int(key): value
                                  for key, value in self.use_radius.items()}
                thresholds = sorted(radius_by_time)
                if thresholds:
                    next_threshold = next((t for t in thresholds if location[0] < t),
                                          thresholds[-1])
                    radius = radius_by_time[next_threshold]
            attrs = {"original_location": location.copy(),
                     "parent_id": parent_id if parent_id >= 0 else None,
                     "track_id": record["track_id"],
                     "value": {"radius": radius, "div_state": div_state}}
            nodes.append(gp.Node(record["cell_id"], location, attrs=attrs))
        batch = gp.Batch()
        batch.graphs[self.points] = gp.Graph(
            nodes, [], gp.GraphSpec(roi=roi.copy()))
        return batch

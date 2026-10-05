"""Small synthetic Biohub stores for the format adapter."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import zarr

from linajea.biohub_io import BiohubImage, read_geff_tracks


SCALE = [1, 1.625, 0.40625, 0.40625]


class BiohubIOTest(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.image_path = Path(self.temp.name) / "sample.zarr"
        self.graph_path = Path(self.temp.name) / "sample.geff"

        image = zarr.open_group(str(self.image_path), mode="w", zarr_format=3)
        image.create_array("0", data=np.arange(4 * 4 * 8 * 8,
                                                dtype=np.uint16).reshape(4, 4, 8, 8))
        image.attrs["multiscales"] = [{
            "axes": [{"name": name} for name in "TZYX"],
            "datasets": [{"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": SCALE}]}]}]
        image.attrs["image_statistics"] = {
            "quantiles": {"0.01": 10, "0.99": 900}}

    def write_graph(self, edges=None):
        if edges is None:
            edges = [[0, 1], [0, 2]]
        graph = zarr.open_group(str(self.graph_path), mode="w", zarr_format=3)
        graph.attrs["geff"] = {
            "directed": True,
            "axes": [{"name": name, "scale": scale}
                     for name, scale in zip("tzyx", SCALE)]}
        nodes = graph.create_group("nodes")
        nodes.create_array("ids", data=np.array([0, 1, 2], dtype=np.int64))
        props = nodes.create_group("props")
        for name, values in zip("tzyx", ([0, 1, 1], [1, 1, 2],
                                         [2, 2, 3], [3, 3, 4])):
            props.create_group(name).create_array(
                "values", data=np.asarray(values, dtype=np.int64))
        graph.create_group("edges").create_array(
            "ids", data=np.asarray(edges, dtype=np.int64).reshape(-1, 2))

    def test_image_metadata_and_lazy_slice(self):
        image = BiohubImage(self.image_path)
        self.assertEqual(image.voxel_size, (1, 52, 13, 13))
        self.assertEqual(image.roi_shape, (4, 208, 104, 104))
        self.assertEqual(image.attrs["stats"]["q001"], 10)
        self.assertEqual(image.read((slice(1, 2), slice(0, 1),
                                     slice(0, 2), slice(0, 2))).shape,
                         (1, 1, 2, 2))

    def test_geff_division_and_world_coordinates(self):
        self.write_graph()
        records, edge_count = read_geff_tracks(
            self.graph_path, (1, 52, 13, 13), (4, 4, 8, 8))
        by_id = {record["cell_id"]: record for record in records}
        self.assertEqual(edge_count, 2)
        self.assertEqual([by_id[i]["parent_id"] for i in (0, 1, 2)],
                         [-1, 0, 0])
        self.assertEqual({record["track_id"] for record in records}, {0})
        self.assertEqual(by_id[2]["z"], 104)
        self.assertEqual(by_id[2]["x"], 52)

    def test_geff_rejects_missing_node_and_duplicate_edge(self):
        for edges, message in (([[0, 99]], "missing node"),
                               ([[0, 1], [0, 1]], "Duplicate GEFF edge")):
            with self.subTest(edges=edges):
                self.write_graph(edges)
                with self.assertRaisesRegex(ValueError, message):
                    read_geff_tracks(self.graph_path, (1, 52, 13, 13))


if __name__ == "__main__":
    unittest.main()

import os
import unittest
import numpy as np
from bev_osm.orienternet_raster import canvas_bounds, load_upstream


class GridContract(unittest.TestCase):
    def test_common_palette_keeps_native_way_priority(self):
        from bev_osm.compare_rasters import common_rgb

        result = {
            "raster": np.array([[[1]], [[3]], [[0]]]),
            "groups": {
                "areas": ["building", "parking", "grass", "park", "forest"],
                "ways": ["road", "building_outline", "fence"],
            },
            "native_rgb": np.array([[[238, 0, 255]]], dtype=np.uint8),
        }
        np.testing.assert_array_equal(common_rgb(result), result["native_rgb"])

    def test_target_extent(self):
        lo, hi = canvas_bounds()
        np.testing.assert_allclose(lo, [-25.25, 0.25])
        np.testing.assert_allclose(hi, [24.75, 50.25])


@unittest.skipUnless(
    os.environ.get("ORIENTERNET_REPO"), "Optional upstream checkout not configured"
)
class NativeRaster(unittest.TestCase):
    def test_metric_centers_camera_and_direction(self):
        u = load_upstream()
        lo, hi = canvas_bounds()
        c = u.Canvas(u.BoundaryBox(lo, hi), 2)
        np.testing.assert_allclose(
            c.to_xy(np.array([[0.0, 0.0], [99.0, 99.0], [50.0, 100.0]])),
            [[-25.0, 50.0], [24.5, 0.5], [0, 0]],
            atol=1e-12,
        )
        self.assertEqual((c.h, c.w), (100, 100))

    def test_native_roads_keep_one_pixel_and_three_categorical_layers(self):
        u = load_upstream()
        lo, hi = canvas_bounds()
        c = u.Canvas(u.BoundaryBox(lo, hi), 2)
        from maploc.osm.data import MapLine, MapArea

        line = MapLine(
            1, "highway:residential", "road", {}, xy=np.array([[0.0, 1.0], [0.0, 49.0]])
        )
        area = MapArea(
            2,
            "building:yes",
            "building",
            {},
            outers=[np.array([[-10, 10], [-5, 10], [-5, 15], [-10, 15], [-10, 10]])],
        )
        masks = u.render_raster_masks([], [line], [area], c)
        raster = u.render_raster_map(masks)
        self.assertEqual(raster.shape, (3, 100, 100))
        self.assertEqual(np.count_nonzero(masks["road"][50]), 1)
        self.assertTrue(np.any(raster[0] == u.Groups.areas.index("building") + 1))
        self.assertTrue(
            np.any(raster[1] == u.Groups.ways.index("building_outline") + 1)
        )
        self.assertFalse(np.any(raster[2]))


if __name__ == "__main__":
    unittest.main()

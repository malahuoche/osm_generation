"""Independent projection/gravity checks; no dataset downloads needed."""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from bev_osm.raw_kitti import level_camera


class RawGeometry(unittest.TestCase):
    def test_camera_leveling_preserves_world_forward_and_gravity(self):
        for heading in [0, 45, 90, 180, 270]:
            h = np.deg2rad(heading)
            f = np.array([np.sin(h), np.cos(h), 0])
            right = np.cross(f, [0, 0, 1])
            level = np.column_stack([right, [0, 0, -1], f])
            tilt = Rotation.from_euler("ZX", [7, -5], degrees=True).as_matrix()
            R = level @ tilt.T
            out, rp, residual = level_camera(R)
            np.testing.assert_allclose(
                R @ Rotation.from_euler("ZX", rp, degrees=True).as_matrix(),
                out,
                atol=1e-12,
            )
            np.testing.assert_allclose(out[:, 1], [0, 0, -1], atol=1e-12)
            self.assertGreater(np.dot(out[:, 2], R[:, 2]), 0.99)
            self.assertAlmostEqual(np.linalg.det(out), 1.0)

    def test_boreas_passive_rotation(self):
        from bev_osm.raw_boreas import boreas_rotation

        r, p, y = 0.2, -0.4, 1.1
        c, s = np.cos, np.sin
        rx = np.array([[1, 0, 0], [0, c(r), s(r)], [0, -s(r), c(r)]])
        ry = np.array([[c(p), 0, -s(p)], [0, 1, 0], [s(p), 0, c(p)]])
        rz = np.array([[c(y), s(y), 0], [-s(y), c(y), 0], [0, 0, 1]])
        np.testing.assert_allclose(boreas_rotation(r, p, y), rx @ ry @ rz, atol=1e-14)

    def test_full_projection_translation(self):
        # Nonzero P[2,3] makes divide-by-diagonal approximation incorrect.
        K = np.array([[700.0, 0, 610], [0, 710, 180], [0, 0, 1.0]])
        t = np.array([0.05, 0.01, 0.003])
        P = np.column_stack([K, K @ t])
        X = np.array([[1.0, 2.0, 10.0], [-2.0, 1.0, 20.0]]).T
        projected = P @ np.vstack([X, np.ones(2)])
        reconstructed = K @ (X + np.linalg.solve(K, P[:, 3])[:, None])
        np.testing.assert_allclose(projected, reconstructed, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

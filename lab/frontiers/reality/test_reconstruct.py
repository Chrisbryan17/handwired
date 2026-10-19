import unittest
import numpy as np
import cv2
from pathlib import Path
import tempfile
from reconstruct import infer_geometry, require_calibration, reconstruct

class SparseGeometryTests(unittest.TestCase):
    def test_two_real_projection_equations_recover_camera_and_sparse_points(self):
        rng = np.random.default_rng(20261019)
        K = require_calibration(920, 1280, 720)
        points = np.column_stack((rng.uniform(-2, 2, 180),
                                  rng.uniform(-1.2, 1.2, 180),
                                  rng.uniform(4, 10, 180)))
        rotation, _ = cv2.Rodrigues(np.array([0.01, 0.12, -0.02]))
        offset = np.array([0.6, 0.02, 0.01])
        def camera(R, t):
            p = (R @ points.T).T + t
            image = (K @ p.T).T
            return image[:, :2] / image[:, 2:]
        first = camera(np.eye(3), np.zeros(3))
        second = camera(rotation, offset)
        first += rng.normal(0, .13, first.shape)
        second += rng.normal(0, .13, second.shape)
        result = infer_geometry(first, second, K)
        self.assertGreater(result["inliers"], 145)
        self.assertLess(result["median_error_px"], .7)
        self.assertGreater(np.dot(result["t"], offset / np.linalg.norm(offset)), .9)
        self.assertLess(np.linalg.norm(result["R"] - rotation), .035)

    def test_reject_unmeasured_focal_and_invalid_matches(self):
        with self.assertRaises(ValueError):
            require_calibration(float("nan"), 1280, 720)
        with self.assertRaises(ValueError):
            infer_geometry(np.zeros((4, 2)), np.zeros((4, 2)), np.eye(3))

    def test_no_fake_pointcloud_if_photo_input_missing(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                reconstruct(Path(d)/"nonexistent_a.png",
                            Path(d)/"nonexistent_b.png", 600,
                            Path(d)/"should-not-exist.ply")
            self.assertFalse((Path(d)/"should-not-exist.ply").exists())

if __name__ == "__main__":
    unittest.main()

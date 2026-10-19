"""Neural Reality: genuine calibrated two-view sparse photogrammetry.

This is a reproducible OpenCV SIFT -> essential matrix -> relative
camera pose -> triangulated sparse PLY pipeline using *actual input images*.
It does not train Gaussian splats, assume metric scale, create fake depths,
or substitute a preloaded 3D scene for reconstruction.

Python >=3.10, pip install -r requirements.txt.
Apache-2.0 OpenCV is an upstream dependency; this program is MIT.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Any
import cv2
import numpy as np


def require_calibration(focal_px: float, width: int, height: int,
                        cx: float | None = None, cy: float | None = None) -> np.ndarray:
    if not math.isfinite(focal_px) or focal_px <= 1 or focal_px > 100_000:
        raise ValueError("Measured or calibrated focal length in pixels is required")
    if width < 64 or height < 64:
        raise ValueError("Image resolution too small for calibrated two-view geometry")
    px = (width - 1) / 2 if cx is None else cx
    py = (height - 1) / 2 if cy is None else cy
    if not all(math.isfinite(v) for v in [px, py]):
        raise ValueError("Principal point must be finite")
    return np.array([[focal_px, 0, px], [0, focal_px, py], [0, 0, 1]], np.float64)


def infer_geometry(points_a: np.ndarray, points_b: np.ndarray, K: np.ndarray,
                   threshold_px: float = 1.4) -> dict[str, Any]:
    """Recover essential matrix/pose with RANSAC and metric-free triangulation.

    Returns camera-to-camera R and translation of arbitrary unit magnitude;
    cannot establish distances in meters without a known baseline.
    """
    pa = np.asarray(points_a, np.float64).reshape(-1, 2)
    pb = np.asarray(points_b, np.float64).reshape(-1, 2)
    if pa.shape != pb.shape or len(pa) < 12 or not np.isfinite(pa).all() or not np.isfinite(pb).all():
        raise ValueError("At least 12 valid two-view image correspondences required")
    if not np.isfinite(K).all() or K.shape != (3, 3) or K[2, 2] != 1:
        raise ValueError("Invalid intrinsic camera calibration")
    E, inlier_mask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, threshold_px)
    if E is None or inlier_mask is None:
        raise ValueError("Essential matrix RANSAC failed; insufficient overlapping parallax")
    if E.shape != (3, 3):
        E = E[:3, :]
    n_pose, R, t, pose_mask = cv2.recoverPose(E, pa, pb, K, mask=inlier_mask)
    if n_pose < 10 or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError("Insufficient geometrically consistent positive-depth points")
    chosen = pose_mask.ravel() > 0
    a = pa[chosen]
    b = pb[chosen]
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    h = cv2.triangulatePoints(P1, P2, a.T, b.T)
    if (np.abs(h[3]) < 1e-12).any():
        raise ValueError("Degenerate homogeneous triangulation")
    xyz = (h[:3] / h[3]).T
    next_xyz = (R @ xyz.T + t).T
    positive = (xyz[:, 2] > 1e-6) & (next_xyz[:, 2] > 1e-6)
    proj1 = (P1 @ np.column_stack((xyz, np.ones(len(xyz)))).T).T
    proj2 = (P2 @ np.column_stack((xyz, np.ones(len(xyz)))).T).T
    projection_valid = (np.abs(proj1[:, 2]) > 1e-9) & (np.abs(proj2[:, 2]) > 1e-9)
    reproj = np.maximum(np.linalg.norm(proj1[:, :2] / proj1[:, 2:] - a, axis=1),
                        np.linalg.norm(proj2[:, :2] / proj2[:, 2:] - b, axis=1))
    keep = positive & projection_valid & np.isfinite(xyz).all(axis=1) & (reproj < 4.0)
    if np.count_nonzero(keep) < 10:
        raise ValueError("Reprojection/cheirality filtering removed nearly all points")
    return {"points": xyz[keep], "matched_indices": np.flatnonzero(chosen)[keep],
            "R": R, "t": t.ravel(), "median_error_px": float(np.median(reproj[keep])),
            "inliers": int(np.count_nonzero(keep)), "matches": len(pa),
            "scale": "arbitrary monocular similarity scale, not meters"}


def reconstruct(image_a: Path, image_b: Path, focal_px: float, out: Path,
                ratio: float = 0.76) -> dict[str, Any]:
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in [image_a, image_b]]
    if any(f is None for f in frames):
        raise ValueError("Unable to open both user-supplied photographs")
    a, b = frames
    assert a is not None and b is not None
    if a.shape[:2] != b.shape[:2]:
        raise ValueError("Both photos must share a calibrated image size; do not silently resize")
    height, width = a.shape[:2]
    K = require_calibration(focal_px, width, height)
    detector = cv2.SIFT_create(nfeatures=9000, contrastThreshold=0.025)
    p1, descriptor_a = detector.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
    p2, descriptor_b = detector.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    if descriptor_a is None or descriptor_b is None or len(p1) < 20 or len(p2) < 20:
        raise ValueError("Not enough image features; use overlapping textured viewpoints")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(descriptor_a, descriptor_b, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < 20:
        raise ValueError("Feature matching failed; insufficient photo overlap")
    a_px = np.array([p1[x.queryIdx].pt for x in good], np.float64)
    b_px = np.array([p2[x.trainIdx].pt for x in good], np.float64)
    result = infer_geometry(a_px, b_px, K)
    xyz = result["points"]
    good_px = a_px[result["matched_indices"]]
    uv = np.rint(good_px).astype(np.int64)
    uv[:, 0] = np.clip(uv[:, 0], 0, width - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, height - 1)
    rgb = a[uv[:, 1], uv[:, 0]][:, ::-1]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii") as fp:
        fp.write("ply\nformat ascii 1.0\n")
        fp.write(f"element vertex {len(xyz)}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(xyz, rgb):
            fp.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                     f"{int(color[0])} {int(color[1])} {int(color[2])}\n")
    metadata = {"source_images": [image_a.name, image_b.name],
                "calibration": {"focal_px": focal_px, "cx": float(K[0, 2]),
                                "cy": float(K[1, 2]), "assumed_zero_distortion": True},
                "reconstruction": {"relative_rotation": result["R"].tolist(),
                    "relative_translation_direction": result["t"].tolist(),
                    "sparse_points": result["inliers"], "initial_matches": result["matches"],
                    "median_reprojection_error_px": result["median_error_px"],
                    "scale": result["scale"]},
                "limitation": "Two-view sparse SfM; no metric measurement, densification, splat training or guaranteed complete geometry"}
    out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_image", type=Path)
    parser.add_argument("second_image", type=Path)
    parser.add_argument("--focal-px", type=float, required=True,
                        help="Calibrated physical camera focal length in pixel units, not guessed FOV")
    parser.add_argument("--out", type=Path, default=Path("reconstruction.ply"))
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.first_image, args.second_image,
                                 args.focal_px, args.out), indent=2))

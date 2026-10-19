# Neural Reality — calibrated two-view reconstruction engine

`reconstruct.py` actually computes **sparse 3D geometry from two
overlapping photographs**. It performs SIFT detection, descriptor
matching with a ratio test, calibrated essential-matrix RANSAC,
relative camera pose recovery, 3D triangulation, positive-depth
cheirality tests, reprojection-error filtering and colored PLY export.

It **does not** display a premade scene as reconstruction, claim
absolute metric distance from two unscaled photographs, infer a
fictional lens calibration, train Gaussian splats, or promise dense
photorealism. The next phase will add real photographs, dataset
provenance, automated COLMAP/multi-view bundle adjustment and
an actual browser viewer.

```sh
python -m pip install -r lab/frontiers/reality/requirements.txt
python lab/frontiers/reality/reconstruct.py photo_01.jpg photo_02.jpg \
    --focal-px 925 --out scene.ply
python -m unittest discover -s lab/frontiers/reality -p 'test_*.py'
```

The focal-pixel value above is **an example** and must be replaced
with the actual calibrated camera focal length; principal point
defaults to the image center and lens distortion is assumed zero.
Translation is fundamentally scale-ambiguous without a measured
baseline or ground control. The JSON sidecar reports these limits.

OpenCV upstream license: Apache-2.0; NumPy upstream: BSD-3-Clause;
the new Handwired source is MIT. User-supplied images are processed
locally, without automatic upload to any server.

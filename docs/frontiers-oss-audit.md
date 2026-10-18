# Eight Frontiers — candidate OSS audit (2026-10-17)
Metadata cross-checked against public upstream repository license declarations.
**This is a candidate audit, not approval of specific vendored files, datasets or model weights.**
All integration decisions require reviewing the exact pinned release, transitive dependencies,
asset/model licenses and NOTICE/attribution requirements.

| Candidate | Declared license | Intended role | Integration decision |
| --- | --- | --- | --- |
| maplibre/maplibre-gl-js | BSD-3-Clause | Earth map rendering | Candidate; attribution and map tiles independently licensed |
| visgl/deck.gl | MIT | Earth visualization layers | Candidate; renderer largely WebGL, distinct from Handwired WebGPU compute |
| OpenStreetMap geographic data | ODbL 1.0 | Earth roads/buildings | Required attribution and ODbL share-alike/Produced Work review |
| InteractiveComputerGraphics/SPlisHSPlasH | MIT | Fluid algorithm reference | Native CPU code; port/independent WGSL implementation |
| InteractiveComputerGraphics/PositionBasedDynamics | MIT | Constraints, fracture/fluids | Native algorithms, not automatic browser acceleration |
| google-research/self-organising-systems | Apache-2.0 | Neural cellular automata | Separate deterministic rules from trained weights |
| hannorein/rebound | GPL-3.0 | N-body scientific oracle | External optional validation ONLY; no source copy into MIT core |
| nerfstudio-project/gsplat | Apache-2.0 | Offline Gaussian training | CUDA backend, requires non-Pages compute environment |
| colmap/colmap | upstream mixed/NOASSERTION metadata | Camera SfM/MVS | Exact binary/code license and components must be reviewed |
| mkkellogg/GaussianSplats3D | MIT | Splat display candidate | Inspect exact dependency licenses before inclusion |
| dimforge/rapier | Apache-2.0 | Collision/contact reference | WASM CPU physics; GPU control must be accounted for separately |
| gkjohnson/three-gpu-pathtracer | MIT | Rendering algorithm reference | Existing implementation is not itself Handwired WebGPU path tracing |
| google-deepmind/mujoco | Apache-2.0 | Articulated physics oracle | CPU/GPU physical backend must be distinguished from GPU policy |
| Handwired | MIT | Compiled numerical circuit backend | Existing fork with inline WGSL generator |

Licensing status is checked only for candidate repository declarations, not
transitive dependencies or redistributable datasets. Do not integrate unknown imagery
or pretend Google Earth / commercial map tiles are open data.

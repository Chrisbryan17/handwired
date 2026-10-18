# GPU 3D contacts: implemented and hardware verified (limited slice)

An Intel UHD Graphics Gen-12LP WebGPU test on 2026-10-18 exercised
real, persistent GPU state in Handwired's 3D linked-cell broad phase and
Jacobi narrow-phase normal impulses, Coulomb kinetic friction, inverse-mass
position correction and semi-implicit gravity prediction.

The new in-kernel impulse accumulation is emitted from the real
`buildContactImpulseCircuit()` Handwired circuit, using
`emitInlineCircuitWgsl`. It is not a renamed handwritten arithmetic
function; the compiler produces the WGSL numerical transport.

## Reproducible hardware evidence

See
[`lab/frontiers/contact-gpu/results/2026-10-18-intel-uhd-circuit.json`](../lab/frontiers/contact-gpu/results/2026-10-18-intel-uhd-circuit.json).

| Real WebGPU scenario | Bodies | CPU contacts | Absolute error | Nonfinite |
| --- | ---: | ---: | ---: | ---: |
| Moving impacts plus anchored bodies | 6 | 2 | 1.862645e-7 | 0 |
| Gravity and sparse spheres | 64 | 1 | 1.490116e-8 | 0 |

The browser confirmed Intel GPU adapter (vendor `intel`, architecture
`gen-12lp`, device `0xa721`) and rejects known software adapters.
The test compares the *full 3D position and velocity arrays* after the
same initial state, settings and exactly one update using an
independently written CPU oracle. No GPU readback is performed inside the
regular solver update; state transfer happens only for diagnostics.

The timed intervals are **single-trial host-observed synchronized
GPU intervals** (5.6 and 4.6 ms) with readback timing separately captured.
These are not median GPU kernel timings, algorithmic scaling claims,
or a CPU/GPU speedup benchmark.

## Real visualization candidate

`lab/frontiers/destruction/demo.html` is the interactive
**Neural Destruction: Impact Laboratory** research preview.
Sphere meshes are generated procedurally and vertex transforms
sample **live GPU-resident, physically updated positions**.
Three independently repeatable scenarios test kinetic impacts, gravity
cascade and moving swarms. Users can adjust gravity, restitution, friction,
camera orbit and zoom, plus run an actual on-demand CPU/GPU state
comparison. No position-to-JavaScript transfer is needed to draw a frame.

The visual browser and mobile quality gates are independently required
before placing it on GitHub Pages. Do not advertise the whole Neural
Destruction project as complete on this evidence.

## Scientific boundaries and open engineering

- Collision shapes: **spheres only**, no triangle meshes, contact
  manifolds, joints, frictional rolling, cohesive structure, fracture
  propagation, finite-element stress tensors or CCD.
- Each GPU step performs exactly one Jacobi contact iteration. Dense
  clusters may exhibit solver stiffness, overlap and energy growth.
- Atomic linked-cell ordering is nondeterministic; the GPU accumulates
  FP32 contacts in a different order from the CPU FP64 per-body accumulator.
- Fixed grid bounds, radius constraint and maximum supported radii
  are enforced. Out-of-domain objects stop interacting with the
  spatial grid; there is no infinite plane boundary.
- No GPU time queries, controlled long-run convergence, external
  benchmark comparisons, reproducible scaling or multi-adapter certification yet.
- Browser mobile verification, camera framing and GPU failure recovery
  need review. A public interactive demo is a preview, not a fracture engine.

## Development verification

```sh
bun test src/runtime/frontiers-contacts.test.ts lab/frontiers/destruction/model.test.ts
bunx tsc -p lab/frontiers/destruction/tsconfig.json
bunx oxlint --type-aware src/runtime/frontiers-contacts-gpu.ts lab/frontiers/destruction
bun scripts/verify-contacts-gpu-browser.ts lab/frontiers/contact-gpu/results/hardware.json
bun scripts/verify-destruction-demo-browser.ts lab/frontiers/destruction/results/preview.json --steps=90 --screenshot
```

The optional private Kaggle GPU runner is
`ops/kaggle/gpu-contacts/kaggle_gpu_contacts.ipynb`. It installs actual
Chrome and attempts WebGPU through a Vulkan device, while requiring the
browser to reject software adapters. Available CUDA GPUs do not imply
working hardware WebGPU. Its output report is not presumed successful
until the Kaggle job's results are inspected.

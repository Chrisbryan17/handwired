# Handwired Visual Lab — Release Engineering and Evidence

Four browser-visible research applications with independently verified Handwired
FP32 circuits, plus an interactive benchmarking dashboard. All applications
have **independent CPU references** and run without neural-network training.

## Public routes (served beneath the repository GitHub Pages path)

- `visual-lab/`: visual gallery and reproducible evidence links.
- `neural-swarm/`: deterministic three-species boids, batched neural steering.
- `neural-maze/`: level-synchronous BFS circuit wavefront, shortest-path tracing.
- `neural-cloth/`: mass-spring cloth with hybrid and GPU-resident physics.
- `neural-fluid/`: five-channel semi-Lagrangian fluid with hybrid and GPU-resident physics.
- `visual-benchmarks/`: independently handwritten JS and WGSL vs Handwired-generated WGSL.

Existing root, `os/`, `learned-gallery/`, and `json-search/` paths are retained.
The site has no server-side GPU dependency: compute executes in the visitor's
WebGPU-capable browser, with the CPU fallback offered by the hybrid modes.

## Engineering architecture

The WebGPU backend baseline is commit
`daeb82eb5fbd8f3ec4e919b777421de952e9aa9e`.
All four experiments originally came from separate feature branches.

`src/runtime/webgpu-inline.ts` emits an inline WGSL function from a real
Handwired `Circuit` (affine, ReGLU, boolean and index read). The full
fluid and cloth GPU solvers call **this compiler-generated function** inside
their dedicated workgroup/stencil compute passes. Their auxiliary shaders for
grid sampling, forces, pressure, collisions and constraints are handwritten.

### Fluid solver

- Five interleaved FP32 channels: velocity X/Y and three RGB dye channels.
- Handwired-generated 5-channel bilinear advection, including GPU-resident
  backtraced 4-tap gathering and optional stirring/dye injection.
- Explicit diffusion, divergence, 20 Jacobi pressure iterations and projection
  in separate compute pipelines.
- Persistent GPU source/target, divergence and pressure buffers; final
  readback is only required for CPU Canvas2D visualization/verification.
- CPU oracle uses the same numerical structure and independently implements
  each step. Hardware rounding differences are reported, never assumed absent.

### Cloth solver

- Mass-spring lattice with eight-neighbor structural/shear forces.
- Handwired-generated two-coordinate Verlet integration inside the spring
  compute kernel.
- Parallel Jacobi position corrections, pinned nodes, mouse dragging,
  spherical obstacle and persistent tear-edge masks.
- Source and target GPU buffers ping-pong; GPU synchronization uses dispatch
  barriers, not intermediate CPU readback.
- CPU oracle uses sequential Gauss-Seidel-like constraints, **not** the same
  relaxation order as GPU Jacobi. Therefore a bounded cloth silhouette error
  is expected; CPU/GPU bitwise equality is not claimed.
- GPU-integrated experiments were tuned after actual screenshot inspection.

### Maze and swarm boundaries

These two experiments are **not** fully GPU-resident global algorithms.
The boid neighbor spatial hash/aggregation and maze frontier-neighbor gathering
are host-side. The actual fixed Handwired update circuit runs on the GPU.
Independent CPU and GPU batches are checked on the exact same source values.

## Verification protocol

```sh
bun install --frozen-lockfile
bun test --timeout 30000 ./lab/visual-swarm/model.test.ts ./lab/visual-maze/model.test.ts ./lab/visual-cloth/model.test.ts ./lab/visual-fluid/model.test.ts
bunx tsc -p .
bunx oxlint --type-aware src/runtime/webgpu-inline.ts lab/visual-swarm lab/visual-maze lab/visual-cloth lab/visual-fluid lab/visual-benchmarks
bun run site:build
```

Real-browser hardware checks, on the authorized Dell with Intel UHD Graphics:

```sh
bun scripts/verify-visual-swarm-browser.ts lab/visual-swarm/results/2026-10-17-release-desktop-final.json 128 --screenshot
bun scripts/verify-visual-maze-browser.ts lab/visual-maze/results/2026-10-17-release-desktop-final.json 128 --screenshot
bun scripts/verify-visual-cloth-browser.ts lab/visual-cloth/results/2026-10-17-release-desktop-final.json 128 --resident --screenshot
bun scripts/verify-visual-fluid-browser.ts lab/visual-fluid/results/2026-10-17-release-desktop-final.json 128 --resident --screenshot
bun scripts/verify-visual-benchmarks-browser.ts lab/visual-benchmarks/results/2026-10-17-release-desktop-final.json 128 --screenshot
bun scripts/verify-resident-cloth-browser.ts lab/visual-cloth/results/2026-10-17-resident-cloth-interactions.json 128
bun scripts/verify-resident-fluid-browser.ts lab/visual-fluid/results/2026-10-17-resident-fluid-interactions.json 128
```

The hardware scripts reject CPU fallback and software WebGPU adapters.
All five browser screenshots and corresponding machine-readable reports
are preserved under the respective experiment `results/` directories.

As of October 17, 2026, the final desktop hardware checks passed:
- Swarm: 128 agents, 20,480 wire comparisons, 0 mismatches,
  maximum error ~5.96e-8.
- Maze: 1,457 cells, 198,152 wire comparisons, 0 mismatches,
  maximum error 0.
- Full GPU fluid: 1,536 cells, 61,440 scalar comparisons,
  0 above tolerance .025, maximum error ~1.91e-6.
- Full GPU cloth: 425 vertices, 6,800 position comparisons,
  0 above tolerance .16, maximum deviation ~.04265.
- Benchmark: both Handwired and independent handwritten WGSL paths
  had **identical outputs** on the two small eight-step tests.

Additional tests validated cloth persistent tears and dragged positions,
and fluid dye injection and finite 33-step trajectories.

## Benchmarks: what the measurements mean

The interactive benchmark times an independent optimized handwritten
JavaScript reference, a Handwired-generated GPU solver, and a
handwritten WGSL GPU solver, with matching numerical algorithms for
the two GPU paths. It records shader/pipeline setup separately, runs
repeated trials, and includes one final readback in each GPU timing.
Values are whole-simulation host-observed latency, not GPU timestamp-query
kernel timings. No universal speedup is claimed.

On Intel UHD Graphics for small batches (24x16 fluid, 15x11 cloth),
GPU often loses to handwritten CPU JavaScript because GPU dispatch overhead
and readback outweigh parallel execution. The WebGPU compiler proof is
numerical/architectural; speed comparisons require larger
calibrated trials and representative adapters.

## Source attribution and publishing

Handwired original upstream: https://github.com/chenglou/handwired

Engineering fork: https://github.com/Chrisbryan17/handwired

Repository source license: MIT. The visual simulations are independently
implemented research demos inspired by Reynolds boids, graph BFS,
spring cloth and Jos Stam-style stable fluids rather than copies of
third-party source. The site uses no pretrained network weights.

No screenshot or static mock-up is represented as an executed kernel.
Machine-readable hardware evidence is mandatory for claims of GPU support.

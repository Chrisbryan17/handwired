# Visual Lab Performance Pass: Maze, Fluid and Cloth

Date: 2026-10-17
Baseline: the main-branch GPU physics implementation preceding this branch.
Branch: `perf/visual-maze-cloth-fluid-20261017`

## Bottlenecks, root causes, and implemented fixes

| Experiment | Previous bottleneck | New fast path | Numerical promise |
| --- | --- | --- | --- |
| Fluid | GPU readback + `onSubmittedWorkDone` and Canvas2D dye drawing on **every frame** | GPU advection, pressure, projection **and GPU bilinear dye rendering** in one submitted command stream. No per-frame synchronization/readback | The **same GPU physics**, checked at the end, not CPU-vs-GPU bitwise parity |
| Cloth | Per-frame readback + CPU rebuilding/drawing a mesh; GPU solver used five Jacobi passes | GPU spring/Verlet/constraint/collision and direct GPU mesh/background rendering; tear masks remain on GPU | The **same GPU physics** and pin constraints, CPU reference uses different relaxation order |
| Maze | CPU neighbor gather, GPU batch upload, GPU readback on each BFS level | GPU gathers the four frontier neighbors, executes the actual Handwired BFS circuit in WGSL and renders GPU-resident visited/frontier/path masks. Readback only at terminal verification | **Exact** visited/frontier/distance parity against independent CPU BFS |

Original legacy GPU/CPU comparison modes remain selectable for research and numerical debugging.
They are intentionally **not** the default visual mode.

The CPU panels in fast Fluid and Cloth are explicitly sampled every four GPU frames,
and are not synchronized, equal-workload speed comparisons. Fast mode displays CPU
submission latency, **not** GPU completion latency. The hardware benchmarks
below synchronize correctly.

GPU compute kernels still embed WGSL generated from the original Handwired
numeric circuit graph; the GPU renderer only consumes computed GPU buffers.

## Intel UHD WebGPU: controlled before/after benchmarks

The comparison uses the **same GPU physics and GPU rendering** for both
execution policies, on the same hardware and the same deterministic initial state.
The baseline calls `await step(); await read()` **every frame**; the optimized
path queues `frame()` calls without blocking and reads the state only once at
the end. Both perform one actual GPU display pass per frame.

| Workload | Frames | Baseline ms/frame | Optimized ms/frame | Acceleration | Max final state difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fluid, 64×40 (2,560 cells) | 120 | 11.7825 | 1.4442 | **8.16×** | **0** |
| Cloth, 33×23 (759 vertices) | 120 | 10.0308 | 0.6500 | **15.43×** | **0** |
| Maze, 47×31 (1,457 cells) | 176 (entire BFS) | 5.6977 | 0.6955 | **8.19×** | **0** |

These are wall-clock host-observed benchmarks with device completion and
a final GPU readback. Compilation and device creation were excluded;
each baseline required per-frame GPU synchronization and GPU buffer mapping.
Measurements are hardware-specific and not pure GPU timestamp-query kernel
measurements. The speedup **does not** mean each physics kernel is 8–15× faster.
It demonstrates the impact of eliminating CPU/GPU roundtrips.

## Correctness gates

- Fluid, 120 GPU frames: 12,800 FP32 state values, zero invalid values,
  nonzero dye preserved, and **zero difference** from the per-frame-readback
  version of the same GPU solver.
- Cloth, 120 GPU frames: 3,036 FP32 state values, no invalid positions,
  pinned positions unchanged, and **zero difference** from the same
  GPU solver with per-frame readback. Tear/pointer interactions are covered
  by the existing resident-cloth hardware tests.
- Maze, 176 GPU BFS levels: full search completed on actual Intel UHD,
  independently reconstructed shortest path length 177, all **4,371**
  visited/frontier/distance state values matched CPU BFS exactly.
  Baseline GPU execution with per-level readback also matched exactly.

Eight-frame real-browser application smoke tests also passed in all three
new fast modes with zero invalid / unmatched state values.

## Reproduce

```sh
bun install --frozen-lockfile
bun test lab/visual-maze/model.test.ts lab/visual-cloth/model.test.ts lab/visual-fluid/model.test.ts
bunx tsc -p .
bunx oxlint --type-aware lab/visual-maze lab/visual-cloth lab/visual-fluid
bun run site:build

# Actual browser execution (real WebGPU adapter required)
bun scripts/verify-visual-maze-browser.ts lab/visual-maze/results/fast-8.json 128 --fast --screenshot
bun scripts/verify-resident-maze-browser.ts lab/visual-maze/results/maze-full.json 128 --screenshot
bun scripts/verify-visual-cloth-browser.ts lab/visual-cloth/results/fast-8.json 128 --fast --screenshot
bun scripts/verify-fast-cloth-browser.ts lab/visual-cloth/results/controlled-120.json 33 --screenshot
bun scripts/verify-visual-fluid-browser.ts lab/visual-fluid/results/fast-8.json 128 --fast --screenshot
bun scripts/verify-fast-fluid-browser.ts lab/visual-fluid/results/controlled-120.json 64 --screenshot
```

Headless tests reject SwiftShader, llvmpipe and browser CPU fallbacks.
The original hybrid/resident verifiers remain available, and the existing
reference implementations are preserved.

## Performance limitations and next engineering gates

1. CPU reference drawing remains on the main thread in fast modes, although
   throttled for Fluid and Cloth. Offloading CPU reference simulation to a
   Worker would further reduce main-thread frame variance.
2. Fluid retains 20 separate Jacobi pressure dispatches. A workgroup-tiled
   iterative pressure solver could reduce dispatch costs, but needs separate
   convergence and divergence tests to retain numerical fidelity.
3. Cloth creates transient bind groups during each step, and its five
   Jacobi passes still invoke separate dispatches. Caching bind groups by
   state/tear parity reduces host overhead further.
4. Browser rAF and display refresh can cap actual screen FPS despite fast
   synchronized GPU timings. Measure achieved rAF cadence separately.
5. Distinct GPU adapters, large grid/mesh sizes, GPU timestamp queries, and
   interactive stress gestures remain valuable cross-device gates.

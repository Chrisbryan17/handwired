# Eight Frontiers — engineering ledger
Baseline: `main` at `9c3ac7d44a470d03536cedaaefea167cd71eb2b1` (2026-10-17).
GitHub Pages baseline includes Swarm, Maze, Cloth, Fluid and a benchmark route.
Do not rewrite prior evidence; require fresh tests for new code.

## Gate semantics
`BACKLOG → ARCHITECTURE → IMPLEMENTATION → CPU VERIFIED → GPU VERIFIED → VISUAL QA → BENCHMARKED → PUBLISHED`.
A gate applies only to functionality demonstrated by evidence. A vertical slice is not the full scope.
A passing CPU test cannot establish GPU hardware support; a browser load cannot establish scientific correctness.

| Workstream | Branch | Gate at initial baseline | Dependency / first vertical slice |
| --- | --- | --- | --- |
| 01 Earth | `feat/frontier-01-earth` | ARCHITECTURE | OSM ingestion and urban road graph; GPU agent state |
| 02 Cataclysm | `feat/frontier-02-cataclysm` | ARCHITECTURE | Shared grid and 3D viewport; actual shallow-water / 3D particle solver |
| 03 Genesis | `feat/frontier-03-genesis` | VISUAL QA (reaction–diffusion vertical slice only) | GPU field update, deterministic CA first; learning/evolution separate |
| 04 Universe | `feat/frontier-04-universe` | VISUAL QA (256-body vertical slice only; full scope open) | Real softened pairwise N-body gravity, Handwired integration |
| 05 Reality | `feat/frontier-05-reality` | ARCHITECTURE | Input/pose pipeline, scene assets, splat viewer; training offline |
| 06 Destruction | `feat/frontier-06-destruction` | ARCHITECTURE | Collision/contact, constraint graph, energy-aware fracture |
| 07 Light | `feat/frontier-07-light` | GPU VERIFIED + visual QA (limited primitive scene, not full scope) | Actual ray-BVH traversal, unbiased estimator, material sampling |
| 08 Machines | `feat/frontier-08-machines` | ARCHITECTURE | Articulated physics reference + GPU batched control |
| Shared platform | `feat/frontiers-platform-20261017` | IMPLEMENTATION | strict device selection, GPU lifecycle, instrumentation, viewer |

## Milestone acceptance
1. Architecture: identified physics assumptions, numerical algorithm, data layout and validation oracle.
2. Implementation: real algorithm runs with meaningful Handwired circuit integration, no fake placeholder dynamics.
3. CPU VERIFIED: deterministic tolerances, independent oracle and invariants pass.
4. GPU VERIFIED: real identified hardware adapter, matching GPU result, device loss, resource limits, no fallback.
5. VISUAL QA: real browser screenshots reviewed; mobile/desktop camera/interaction validated.
6. BENCHMARKED: GPU synchronized timings, transfer, setup, memory and equivalent CPU baseline.
7. PUBLISHED: Pages route or approved backend, verified public deployment and reproducibility.
First real-hardware QA captured two genuine execution states: 256-body Universe after 281 GPU frames and 192²-cell Genesis after 1,272 GPU iterations, both on Intel UHD Gen-12LP. The corresponding one-step CPU/GPU differential checks passed with maximum absolute error 5.960e-8, but long-run numerical validation and comparable performance benchmarking remain open. The original scene screenshots are preserved under `lab/frontiers/{genesis,universe}/results/2026-10-17-evolved-hardware.png`. The original 256-body slice passed CPU tests and actual Intel UHD GPU browser differential; Barnes–Hut, galaxy morphology/scaling,
conservation statistics and million-body performance remain future gates.

## Blockers and open engineering debt
- Kaggle CLI availability on Dell does not imply Kaggle supplies Chromium WebGPU hardware; verify adapter and backend inside actual remote browser process.
- Static GitHub Pages cannot provide server-side multi-view training or COLMAP processing.
- OSM tiles, elevation, buildings and traffic require real separately licensed data, verified source and attribution; never synthesize them as measured Kingston.
- Native engine references may not be portable to browser WGSL without substantial reimplementation.
- Need amortized pipelines, shared spatial index, timestamp-query support checks, GPU memory accounting, window resize/device loss, and WebGPU browser screenshot artifacts for every new route.
- REBOUND GPL-3.0 reference cannot be source-integrated into MIT components without license review.
- GPU rendering != GPU simulation; controller inference != rigid-body physics != learning.

## Light research slice verification (2026-10-18)
- Physically computed diffuse/reflected/refracted Monte Carlo light transport, Handwired RGB throughput circuit, actual progressive GPU accumulation.
- Intel UHD Gen-12LP WebGPU 12-spp radiance smoke (19,200 pixels) and real 200-spp desktop / 59-spp mobile screenshots. Not a claim of photorealism or full-scene Monte Carlo parity.
- Independent CPU intersection and path reference; new GPU/CPU deterministic ray traversal parity is required to pass before merging this gate.
- Outstanding: BVH, multiple importance sampling, GGX materials, unbiased-transport estimator analysis, multi-seed convergence and performance, cross-adapter verification.

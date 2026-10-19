# Eight Frontiers — living engineering ledger
Verified checkpoint: 2026-10-18 Jamaica time; `main` at
`e313d9f084d62d0beca1fb1dc28d2eb072e6ecab`.
Source of truth for a workstream is its exact feature-branch commit
and attached real test evidence, not historical status notes.

## Kanban gate semantics

`BACKLOG → ARCHITECTURE → IMPLEMENTATION → CPU VERIFIED → GPU VERIFIED → VISUAL QA → BENCHMARKED → PUBLISHED`

Gates below apply to **specific functional slices** unless stated;
no whole frontier is finished. A real browser page is not proof of
scientific accuracy or GPU hardware execution. NVIDIA native
`wgpu-native/Vulkan` compute is hardware execution but not browser
WebGPU certification.

| Workstream | Verified vertical-slice gate | Production/publication boundary | Next hard blocker |
| --- | --- | --- | --- |
| **01 Neural Earth** | ARCHITECTURE | No validated Kingston 3D agent twin yet | Real licensed road/building/elevation data, routing and GPU agents |
| **02 Neural Cataclysm** | **GPU VERIFIED (native NVIDIA Tesla T4; WCSPH slice)** | Interactive 3D candidate on `feat/frontier-02-cataclysm`; *not published* | Real Chromium hardware screenshot QA, stability, surface reconstruction and tsunami coast model |
| **03 Neural Genesis** | PUBLISHED (limited cellular/reaction–diffusion slice) | [Browser](https://chrisbryan17.github.io/handwired/neural-genesis/) exists via Pages route; morphology/evolution are incomplete | True mutation/selection/training beyond deterministic field rules |
| **04 Neural Universe** | PUBLISHED (limited 256-body pairwise gravity slice) | [Browser](https://chrisbryan17.github.io/handwired/neural-universe/) exists via Pages route; no millions-body gravity claim | Barnes–Hut, energy checks, scalability and colliding galaxy complexity |
| **05 Neural Reality** | ARCHITECTURE | No reproducible input-to-reconstruction pipeline yet | COLMAP/SfM workflow, licensed images, splat training and viewer |
| **06 Neural Destruction** | **GPU VERIFIED (native NVIDIA Tesla T4; bonded fracture slice)** | Interactive candidate in `feat/frontier-06-destruction`; *not published* | Actual browser visuals/QA, structural stability, rigid-shape contact and fracture/debris coupling |
| **07 Neural Light** | PUBLISHED (limited physically traced scene) | [Browser](https://chrisbryan17.github.io/handwired/neural-light/) and desktop/mobile hardware evidence; full PBR scope open | Acceleration BVH/MIS/GGX, convergence and complex objects |
| **08 Neural Machines** | CPU VERIFIED (2DOF leg slice); GPU candidate committed | `feat/frontier-08-machines`, private GPU test submitted; *not published* | Physical T4 differential run, free-floating articulated contact and actual gait training |
| **Shared GPU physics** | GPU VERIFIED (Intel Gen-12LP sphere contacts + 3D grid) | Merged by [PR #14](https://github.com/Chrisbryan17/handwired/pull/14) | Long-run stability and dense-contact convergence |

## Current verifiable checkpoints

- **Cataclysm:** Actual NVIDIA Tesla T4 native Vulkan execution of
  GPU linked-cell SPH, density, Handwired-generated pressure,
  viscosity and boundary dynamics. 64/80/120-body independent
  CPU comparisons passed; a separate 18,432-particle, 32-step
  GPU-only conservation test passed. Original raw
  [small/large report](../lab/frontiers/cataclysm/results/2026-10-18-tesla-t4-18432-sph-stress.json)
  lives in the feature branch, not `main`.
  That large run does not prove per-particle CPU parity at 18,432.
- **Destruction:** Actual NVIDIA Tesla T4 native Vulkan verified
  36-node/107-bond graph with physical 5-bond critical fracture,
  zero bond-state mismatches and small velocity residuals.
  [Evidence](../lab/frontiers/destruction/results/2026-10-18-tesla-t4-native-fracture.json)
  is on the feature branch only; full 3D browser visual gate remains open.
- **Shared contact:** Actual Intel UHD Gen-12LP browser WebGPU
  one-step differential passed for 6 and 64 spheres and compiler-generated
  Handwired impulse arithmetic. This source and
  [evidence](../lab/frontiers/contact-gpu/results/2026-10-18-intel-uhd-circuit.json)
  are now on `main`.
- **Machines:** CPU Lagrangian two-joint solver with joint inertia
  coupling, ground penalty force and deterministic Handwired PD
  controller passed focused tests. GPU-batched kernel and private
  Tesla T4 runner committed; *no hardware result claimed yet*.
- **Public:** `scripts/demo-site-pages.ts` registers Neural Genesis,
  Neural Universe, Neural Light and the existing four Visual Lab
  demonstrations. The three new public routes represent limited
  research implementations, not completed frontiers.

## Active dependencies and blockers

1. Kaggle Linux has physical NVIDIA Vulkan but Chrome's WebGPU
   selects SwiftShader. Native computational verification is valid;
   browser execution remains independently unverified.
2. Dell's memory pressure makes sustained local Chromium/GPU visual
   testing undesirable. Use small guarded browser tests only when
   resources permit, or an authorized suitable remote GPU browser.
3. Candidate 3D fluid/fracture visuals require real captured images,
   controls, desktop/mobile QA, meaningful synchronized benchmarks,
   deployment route checks, and source attribution before merging
   their public demo features.
4. Earth requires licensed OSM/terrain data; Reality needs a real
   user-input SfM/reconstruction/training path; Machines needs
   free-body physics and training. None is replaceable with a cosmetic viewer.
5. NVIDIA timing figures in native reports include setup, submission
   and diagnostic readback, not pure timestamp-query compute latency.
   No CPU/GPU speedup ratio is established by those numbers.

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

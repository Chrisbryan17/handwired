# Neural Cataclysm — 3D weakly compressible SPH research slice

**State:** CPU-verified and **native NVIDIA Tesla T4 Vulkan/GPU verified**
for the exact WebGPU WGSL linked-cell, pressure and viscosity solver. Real
interactive browser implementation is in the feature branch.
**Browser execution/visual QA on hardware: OPEN.**

This is not yet a coastal tsunami simulator or physically reconstructed
free-surface renderer. The actual particle positions are computed by
fluid dynamics, never by keyed oscillatory vertices or synthetic
wave animations.

## Numerical model and independent oracle

`model.ts` contains a handwritten O(N²) CPU implementation:

- Mass-weighted compact-support 3D poly6 density estimates.
- Spiky radial pressure gradients for pairwise repulsion.
- Symmetric pressure-force term based on both particle pressures and
  densities, plus viscous pairwise velocity diffusion.
- Weakly compressible pressure equation of state
  `p = max(0, stiffness * (density - restDensity))`.
- Semi-implicit Euler gravity/pressure/viscosity integration.
- Reflecting tank boundaries with configurable coefficient of restitution.
- All mass stored in particle `vec4.w`, conserved across state updates.

A real Handwired `Circuit` builds the signed pressure multiplier used
by the GPU module's inlined compiler-generated WGSL. The kernel's
surrounding neighborhood density and force reductions are handwritten
WGSL, not automatically generated neural-network code.

## GPU implementation

`solver.ts` runs three actual GPU compute phases:

1. **Rebuild a GPU-linked 3D cell index** using atomic cell-heads and
   persistent per-particle linked lists, without a redundant neighbor
   count pass.
2. Compute local kernel densities and equation-of-state pressure using
   the Handwired-generated circuit. All neighboring positions are
   read from GPU-resident buffers.
3. Compute SPH pressure forces, viscous acceleration, gravity and tank
   collisions. Update next-step state entirely on GPU.

The next-step particle state is copied device-to-device. No CPU readback
is performed by `step()`. Diagnostics `read()` and
`readDensityPressure()` intentionally synchronize when explicitly
requested for correctness testing.

## Verified NVIDIA hardware execution (2026-10-18)

The full, actual 3D SPH WGSL modules were exported unchanged from
Handwired source and executed through `wgpu-native` using a
**physical NVIDIA Tesla T4 discrete GPU**, driver 580.178.04, Vulkan
backend. The actual adapter identity was inspected from `wgpu-native`,
not inferred from CUDA presence. Software adapters were rejected.

An independent handwritten CPU `model.ts` oracle supplied positions,
velocities, density and pressure baselines; results were compared after
synchronized device-to-host readback:

| Scenario | N | GPU steps | Max state error | Max pressure error | Nonfinite | Mass loss |
|---|---:|---:|---:|---:|---:|---:|
| Pressure | 64 | 1 | 2.980e-8 | 4.196e-5 | 0 | 0 |
| Viscous flow | 80 | 4 | 1.192e-7 | 3.052e-5 | 0 | 0 |
| Gravity flow | 120 | 8 | 2.384e-7 | 2.289e-5 | 0 | 0 |

Permanent machine-readable evidence:
[`results/2026-10-18-tesla-t4-native-full-sph.json`](../lab/frontiers/cataclysm/results/2026-10-18-tesla-t4-native-full-sph.json).
The separately compiled Handwired pressure Circuit also passed
**256/256 exact comparisons** (max error 0.0):
[`pressure evidence`](../lab/frontiers/cataclysm/results/2026-10-18-tesla-t4-native-pressure.json).

These are actual NVIDIA GPU compute results, **not Chromium
browser WebGPU tests**. The prior Kaggle Chrome experiment correctly
rejected SwiftShader even when native Vulkan could see Tesla T4s.
The timings in the reports include shader setup, CPU submission and/or
readback. Do not misrepresent them as isolated GPU kernel time or
CPU/GPU speedup.

## Large-scale native SPH stress — 2026-10-18

The same Vulkan Tesla T4 GPU kernel suite subsequently ran **18,432
real fluid particles for 32 complete simulation steps**. The large case
was *not* compared body-for-body against the O(N²) CPU oracle; it was
separately evaluated for conserved mass, finite velocities/positions
and tank bounds. All these invariants passed:

- Mass loss: **0**
- Nonfinite values: **0**
- Boundary violations: **0**
- Recorded maximum speed: approximately **0.4287 model units/s**

Raw evidence: [Tesla T4 SPH stress JSON](../lab/frontiers/cataclysm/results/2026-10-18-tesla-t4-18432-sph-stress.json).
The test recorded approximately 97.3 ms for 32 steps **including
diagnostic readbacks** and GPU submission. This is **not** a per-kernel
GPU timestamp result or a CPU-versus-GPU throughput speedup claim.
The existing 64-/80-/120-particle CPU parity cases passed again.
This demonstrates a bounded fluid computation, not a tsunami at
physical city scale.

## Browser research laboratory

`demo.html` offers three deterministic physical scenarios:
3D dam-break-like particle release, counter-propagating currents
and a falling droplet block. Real GPU storage-buffer positions drive
a lit orbitable particle renderer. Gravity, viscosity and pressure
stiffness are interactive; a button compares one live WebGPU step
against the independent CPU solver.

The spherical particles are **visual markers**, not a signed-distance
surface, particle meshing or physically correct water transmission.
No offline/rendered animation is used as evidence of fluid computation.

## Verification

```sh
bun test lab/frontiers/cataclysm
bunx tsc -p lab/frontiers/cataclysm/tsconfig.json
bunx oxlint --type-aware lab/frontiers/cataclysm
bun build lab/frontiers/cataclysm/demo.html --outdir /tmp/cataclysm
bun scripts/verify-cataclysm-browser.ts \
    lab/frontiers/cataclysm/results/local-intel-webgpu.json --screenshot
```

The real GPU test checks 64-, 80- and 120-particle scenarios,
1-, 4- and 8-step simulation evolutions, independently computed
density/pressure fields, positions, velocities, conserved mass,
bounded numerical tolerances and actual GPU adapter identification.
CPU fallback or SwiftShader must not count as hardware verification.

## Remaining work

- Chromium/WebGPU execution on identified physical GPU and real screenshots.
- Further long-running stability, density-energy diagnostics and more complex particle distributions.
- Spatial locality improvements (contiguous/sorted cells).
- Stabilization, CFL-based adaptive stepping, WCSPH tensile instability
  diagnostics and strong-pressure energy tests.
- Rigid solid/particle coupling, buoyancy, debris, erosion,
  terrain bathymetry, tsunami boundary inflow/outflow and
  surface reconstruction.
- Camera/UI mobile QA, actual browser screenshots, reproducible
  latency vs the independent CPU oracle, multiseed and multiadapter
  verification, and final public publication.

These limits are explicitly not concealed by the fact that the 3D
GPU solver code compiles or its CPU reference tests pass.

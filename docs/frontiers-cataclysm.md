# Neural Cataclysm — 3D weakly compressible SPH research slice

**State:** CPU-verified 3D fluid solver, WebGPU linked-cell SPH compute
implementation and interactive GPU-buffer-backed particle laboratory.
**Hardware GPU numerical differential and visual QA: OPEN.**

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

- Hardware-executed GPU/CPU differential tests and evidence capture.
- GPU memory stress and larger particle counts with conservation checks.
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

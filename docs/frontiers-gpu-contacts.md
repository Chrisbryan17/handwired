# Experimental GPU Sphere Contacts — hardware gate open

This branch implements a genuine WebGPU solver that couples the shared 3D
atomic linked-cell spatial index to narrow-phase sphere impulses and
semi-implicit Euler prediction. It is a **research implementation awaiting
hardware numerical parity**; it is not yet a verified GPU-accelerated
rigid-body simulator.

## Actual computation

A single state update stays on GPU:
1. Predict dynamic body positions/velocities (gravity and semi-implicit Euler).
2. GPU-to-GPU copy predicted positions into spatial index positions;
   no CPU readback.
3. Rebuild atomic cell lists and query local 3D grid neighbors.
4. Compute a per-body Jacobi contact response (normal restitution,
   Coulomb kinetic friction and inverse-mass-weighted penetration correction).
5. GPU-to-GPU copy corrected position/velocity state for the next frame.

This implements at most **one Jacobi contact iteration per update**.
The separate CPU oracle accepts multiple iterations but the GPU solver
explicitly rejects settings above one until parity for multiple passes
is demonstrated.

Constraints: the grid's search radius must be at least twice the maximum
sphere radius, and no greater than cell size. Arbitrary particle radii
beyond this bound are rejected. Point domain boundaries are fixed;
there are no external walls, joints, continuous collision detection,
fracture or surface meshes. The grid contact traversal uses atomic-linked
heads and may differ in FP32 summation order from the CPU oracle.

## Required hardware verification

```sh
bun test src/runtime/frontiers-contacts.test.ts
bunx tsc -p lab/frontiers/contact-gpu/tsconfig.json
bunx oxlint --type-aware src/runtime/frontiers-contacts-gpu.ts lab/frontiers/contact-gpu
bun scripts/verify-contacts-gpu-browser.ts lab/frontiers/contact-gpu/results/hardware.json
```

The last command must identify a real hardware adapter and compare
predicted positions, impulse velocities, penetration correction and friction
against `stepSphereContactsCpu`. Compilation alone, headless SwiftShader,
or a CPU reference result do **not** constitute GPU verification.

Do not merge or advertise this as GPU-verified contact physics until a
real hardware run passes, numerical conservation checks are reviewed,
and visual/contact stability tests are recorded. This module has **no**
public destructive scene or claim of structural engineering fidelity.

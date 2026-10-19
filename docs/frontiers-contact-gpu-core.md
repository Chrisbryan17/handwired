# Handwired GPU sphere contacts — verified shared runtime

A 3D sphere-contact research solver is implemented in
`src/runtime/frontiers-contacts-gpu.ts`. It reuses the shared
GPU atomic linked-cell spatial grid and genuine Handwired-compiled
WGSL circuit impulse arithmetic. Semi-implicit position/velocity
prediction, restitution, Coulomb kinetic friction and penetration
correction execute on GPU-resident buffers. Readback is diagnostic
rather than per-frame.

## Verified independent hardware evidence

The [Intel UHD Chrome hardware probe](../lab/frontiers/contact-gpu/results/2026-10-18-intel-uhd-circuit.json)
reported Intel Gen-12LP device `0xa721` with
`hardware=true`. Complete output state was compared against an
independently handwritten CPU contact oracle after synchronization.

| Scene | Bodies | Physics steps | Maximum position/velocity error |
|---|---:|---:|---:|
| Multi-contact, anchored body | 6 | 1 | 1.862645e-7 |
| Gravity and sparse contacts | 64 | 1 | 1.490116e-8 |

Both passed and had zero nonfinite values. The single-run synchronized
timings include host activity and are **not** GPU timestamp query times
or speedup claims. The result establishes only those two one-step
comparison cases; long-run stability, dense crowd broad-phase
scaling, and independent adapters still need testing.

## Reproduction and boundaries

```sh
bun test src/runtime/frontiers-contacts.test.ts
bunx tsc -p .
bunx oxlint --type-aware src/runtime/frontiers-contacts-gpu.ts
bun scripts/verify-contacts-gpu-browser.ts \
  lab/frontiers/contact-gpu/results/hardware.json --screenshot
```

The browser script rejects software fallback (SwiftShader, llvmpipe).
Native Vulkan compute on NVIDIA T4 is separately labeled and is
not interchangeable with browser WebGPU evidence.

This is **sphere-only** narrow-phase contact with a single Jacobi
impulse pass. It is not continuous collision detection, joints,
bonded structural fracture, arbitrary triangle collision, or a
full robotics simulator. No destructive 3D visualization or public
frontier route is claimed in this reusable-core merge.

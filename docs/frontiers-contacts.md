# Handwired Frontiers — contact dynamics CPU reference

This workstream supplies a deterministic handwritten **narrow-phase
sphere-contact oracle** for Neural Destruction and Neural Machines,
and a future GPU contact solver. It is not an arbitrary-mesh rigid-body
engine or a GPU physics accomplishment.

## Representations and dynamics

- Each sphere: positions `[x,y,z,radius]`, velocities
  `[vx,vy,vz,inverseMass]`, all FP32. Inverse mass zero anchors a sphere.
- Semi-implicit Euler gravity/drift applied only to bodies with nonzero
  inverse mass.
- For each pair whose spheres overlap, independently compute:
  - Unit normal and penetration depth.
  - Relative normal speed and approaching-contact impulse
    `J_n = -(1+e) (v_rel·n) / (mA^-1+mB^-1)`.
  - Coulomb **kinetic** tangential impulse, bounded by `μ J_n`.
  - Inverse-mass-weighted Baumgarte-style position correction with
    configurable slop and correction percentage.
- Jacobi-style pair impulse accumulation, followed by state update.
  Multiple iterations reduce severe overlap; contact order does not
  directly update the current velocities in the inner pair loop.
- An independent `Circuit` builds a signed momentum-kick scalar
  multiplier for eventual in-kernel Handwired-compiled WGSL.

## Constraints and limitations

This solver's broad phase is O(N²) and caps the CPU reference at
10,000 spheres. It is **not coupled** to the new GPU spatial linked-cell
index yet and it does **not** run contact dynamics on the GPU. No continuous
collision detection, persistent manifolds, joint motors, fracture,
stress/material modeling, particle-fluids, aerodynamic forces, or
full robotic articulation. A zero-separation normal is resolved by
a deterministic coordinate axis, not a geometrically unique answer.

Conservation of momentum/kinetic energy is tested for idealized equal-mass
elastic head-on impact with zero friction and no gravity. Friction should
decrease tangential relative energy; immovable bodies do not drift.
The normal impulse is a constrained rigid-contact approximation and not a
material-specific continuum-mechanics stress estimate.

## Focused tests

```sh
bun test src/runtime/frontiers-contacts.test.ts
bunx tsc -p .
bunx oxlint --type-aware src/runtime/frontiers-contacts.ts
```

Test cases cover mass and momentum accounting, elastic restitution,
static anchor, friction, penetration correction, deterministic update,
and Handwired arithmetic equivalence.

## Next GPU milestone

1. Move sphere centers and velocities into persistent storage buffers
   while preserving exact CPU matching initial conditions.
2. Reuse spatial-grid GPU cell-head and next-list buffers for candidate
   broad-phase, and issue a second narrow-phase compute dispatch.
3. Make contacts race-free through per-body Jacobi accumulators; add
   conservation diagnostics, solver iteration convergence and stability
   tests, plus cross-adapter scrutiny.
4. Only after actual GPU tests claim GPU rigid-body computation.

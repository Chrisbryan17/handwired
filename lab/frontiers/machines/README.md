# Neural Machines — pinned planar leg physical reference

First research vertical slice: an **exact two-revolute-link planar
dynamical model** pinned at a fixed hip, with two point masses,
independently computed coupled Euler–Lagrange inertia, Coriolis terms,
gravity, capped PD motor torque, joint damping and penalty-based
normal/tangential foot contact.

**Current evidence: CPU reference and tests only.** GPU implementation,
hardware parity, visual locomotion, articulated free-body dynamics
and policy optimization have not yet been verified.

## Differential equations

For a hip angle `q1` and relative knee angle `q2`,
`M(q) * qdd + C(q,qd) + G(q) = tau_motor + J(q)^T F_ground - D*qd`.

The independent mass matrix uses concentrated endpoint masses,
not a continuous rigid-link moment of inertia. The analytical
determinant at `q2=0` is `m1*m2*l1²*l2²` (strictly positive with
nonzero masses/lengths). Full 2x2 matrix inversion includes
off-diagonal inertial coupling and nonzero Coriolis terms.

Foot ground contact uses a **penalty spring/damper**, penetration
limited normal reaction and bounded kinetic tangential damping.
It is not a complementarity-based rigid contact constraint and
can permit transient penetration. Hip is spatially pinned, so this
does not simulate a free-moving walking robot.

## Handwired boundary

`buildLegMotorCircuit()` emits a two-term PD torque computation
via real Handwired `Circuit` operations. The motor is an explicitly
programmed feedback controller, **not a neural network trained
with reinforcement learning**. GPU controller integration and
larger parallel environment throughput are separate acceptance gates.

## Focused CPU reference

```sh
bun test lab/frontiers/machines/model.test.ts
bunx tsc -p lab/frontiers/machines/tsconfig.json
bunx oxlint --type-aware lab/frontiers/machines
```

Tests compare independent motor arithmetic, physical inertia
determinants, true contact penetration, friction bounds,
passive mechanical energy, seeded deterministic ensembles,
and rejection of invalid physical parameters.

## Next gates

Actual GPU-batched 2DOF evolution, CPU/GPU state parity
on independent initial conditions, 4,096+ controller instances,
real hardware adapter and timing evidence, 3D visualization,
free-body motion, continuous contact, articulated multi-leg
constraints, gait training and reproducible replay.

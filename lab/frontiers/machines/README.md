# Neural Machines — pinned planar leg physical reference

First research vertical slice: an **exact two-revolute-link planar
dynamical model** pinned at a fixed hip, with two point masses,
independently computed coupled Euler–Lagrange inertia, Coriolis terms,
gravity, capped PD motor torque, joint damping and penalty-based
normal/tangential foot contact.

**Current evidence: independent CPU tests and real NVIDIA Tesla T4
native Vulkan verification of actual GPU 2DOF mechanics and Handwired
motor circuits.** Browser hardware rendering, full locomotion,
free-body physics and learning remain unverified.

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

## GPU batched 2DOF model added (hardware acceptance pending)

`gpu.ts` uses the same exact articulated two-link mass matrix and
contact penalty model as the independent CPU solver, compiled into
a real WGSL compute pass. A Handwired-generated PD motor circuit
is invoked for each joint on every simulated robot. GPU storage
is ping-ponged between steps; no CPU uploads/readbacks during
regular simulation. The per-environment calculation is parallel,
not a claim of locomotion training or free-body multilegged physics.

Strict TypeScript/lint and the native hardware CPU/GPU
differential remain required gates. Compilation alone is not
a real hardware pass.

## Physical NVIDIA GPU verification — 2026-10-18 (Jamaica)

The actual Handwired-generated controller WGSL and 2DOF dynamics
were dispatched through `wgpu-native` on a **physical Tesla T4
Vulkan device** (driver 580.178.04). Full joint-position and
joint-velocity arrays were compared with independent CPU dynamics
after synchronization:

| Scenario | Legs | Physics updates | Maximum absolute error | Invalid values |
|---|---:|---:|---:|---:|
| Ground contact | 64 | 1 | 7.451e-8 | 0 |
| Feedback control | 256 | 24 | 4.172e-7 | 0 |
| Batched dynamics | 4,096 | 64 | 1.132e-6 | 0 |

All pass. The original machine-readable
[hardware evidence](results/2026-10-18-tesla-t4-4096-articulated.json)
records the adapter, tolerance, shader hash and CPU-parity metrics.
The elapsed intervals include setup/host command submission and
readback and must **not** be described as pure GPU kernel timings.

This validates real batched GPU physical computation for those
scenarios, **not Chromium browser GPU execution** and not a trained
robotics policy. Nothing is yet publicly deployed.

# Neural Destruction — cohesive fracture computational research

Current integration gate: **CPU VERIFIED** plus GPU structural implementation,
TypeScript and bundler checks. The structural GPU-specific numerical and
browser visual verification gates are **OPEN**. No public fracture demo is
represented as complete or hardware verified until those pass.

## Computational basis

A discrete three-dimensional bridge of nodes and cohesive bonds is
integrated using a Hookean axial spring model and a Kelvin–Voigt
velocity-proportional damper:

```text
e_ij = (x_j - x_i) / ||x_j - x_i||
extension = ||x_j - x_i|| - restLength
relativeSpeed = dot(v_j - v_i, e_ij)
F_ij = [stiffness * extension + damping * relativeSpeed] * e_ij
```

The force on node `j` is equal and opposite to `i`. Anchored
nodes (inverse mass 0) do not integrate. Dynamic nodes use
semi-implicit Euler gravity and nodal forces. Broken links contribute
no subsequent force.

For accumulated strain damage:

```text
strain = |extension| / restLength
damage += dt * damageRate * (strain - yieldStrain) /
          (criticalStrain - yieldStrain)    [only above yield]
failure = strain >= criticalStrain OR damage >= 1
```

Damage is monotonic and fracture is irreversible. Geometry disappears
from the viewer when *computed GPU bond state* declares it broken. This
is not a keyed timeline, a texture fracture animation, or prestaged
cascading debris.

## CPU baseline

`fracture-model.ts` is a separately handwritten CPU oracle with FP64
nodal force accumulation, FP32 state storage and explicit damage checks.
The analytical tests validate one-dimensional spring acceleration,
conservation of momentum in symmetric elastic loading, irreversible
threshold failure, accumulated damage, external impulses, anchored
supports, topology validation and time-step constraints.

## WebGPU implementation

`fracture-gpu.ts` has two GPU compute stages:

1. One thread per bond computes physical strain, viscous/elastic axial
   force and irreversible bond damage. The active/broken state persists
   in WebGPU storage.
2. One thread per node gathers adjacent bond forces from a fixed,
   precomputed **CSR sparse graph**. GPU acceleration, velocities and
   positions are integrated with real WGSL emitted from a Handwired
   numerical Circuit. Node and bond state remains GPU-resident,
   including subsequent steps.

External user-triggered impacts run a separate GPU compute pass that
updates the targeted node's velocity buffer; they are not scripted
collapse animations or CPU frame advancement.

`fracture-renderer.ts` generates lit cylindrical beam geometry from
current GPU node positions and bond state with GPU depth testing and an
orbit camera. Strain damage changes beam color; severed bonds are no
longer drawn. The browser experience `fracture-demo.html` has
material-strength and gravity controls, several mesh resolutions and a
live CPU/GPU state comparison button.

## Hardware acceptance

```sh
bun test lab/frontiers/destruction/fracture-model.test.ts
bunx tsc -p lab/frontiers/destruction/tsconfig.json
bunx oxlint --type-aware lab/frontiers/destruction
bun scripts/verify-fracture-browser.ts lab/frontiers/destruction/results/fracture.json --screenshot
```

A hardware pass must identify a real WebGPU adapter, compare
actual GPU positions/velocities and per-bond failure flags against the
CPU oracle for elastic loading, a deliberately strained break and
multiple steps, and reject software fallback. The live browser
visualization then requires screenshot inspection at desktop/mobile
viewports, control manipulation and verification of broken geometry.

**Important limitations:** This is not a finite-element method,
calibrated continuum damage model, surface/volume crack propagation,
material-specific engineering standard, mesh self-contact or debris
dynamics. Spring stiffness and damage thresholds are intentionally
synthetic dimensionless parameters. Such capabilities remain outstanding
before the user's full Neural Destruction scope can be called complete.

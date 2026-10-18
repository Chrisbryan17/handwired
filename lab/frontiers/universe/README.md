# Neural Universe — first functional research slice

Real GPU-resident softened Newtonian direct N-body with two initialized rotating disks.
Each simulated body contributes gravitational force to every other body at O(N²)
time complexity. State remains in two alternating GPU storage buffers and is
rendered via GPU instanced billboards. The compute kernel includes
`handwiredKickDrift`, generated from the actual Handwired numerical Circuit.
Uncoupled 3D star billboard graphics are **not** claimed to be path tracing.

## What is genuinely implemented
- 128, 256, 512 and 1024 selectable gravitational bodies, including massive cores.
- Pairwise softened gravitation and fixed-step symplectic Euler integration.
- Real shader-generated Handwired FP32 arithmetic in the integration pass.
- Interactive rotation, zoom, gravity and softening; reset and pause.
- CPU oracle independent of Handwired, momentum/energy diagnostics.
- Explicit synchronized GPU comparison button (does not silently read back per frame).
- Browser-only GPU graphics and compute, no CPU fallback.

## Scientific boundaries
This is not Barnes–Hut or FMM, is not million-body gravity,
is not galaxy-formation cosmology, and is not yet long-run energy validated.
Star initial conditions are synthetic, not astronomical measurements.
The GPU solver uses fast f32 inverse square root while the CPU reference uses
IEEE JavaScript double precision followed by float32 state rounding. Agreement
on one step does not establish error growth after a thousand steps.

## Run and test

```sh
bun install --frozen-lockfile
bun test lab/frontiers/universe/model.test.ts
bunx tsc -p lab/frontiers/universe/tsconfig.json
bunx oxlint --type-aware src/runtime/frontiers-device.ts lab/frontiers/universe
bun run site:build
bun scripts/verify-universe-browser.ts lab/frontiers/universe/results/local.json --screenshot
```

The last command requires actual browser WebGPU, identifies adapter and rejects
common software implementations. A GPU screenshot is stored locally by the
verifier. The 2026-10-17 hardware report JSON in this branch was reproduced
verbatim from the successful Dell verifier stdout; the PNG is not yet committed.

## First verified hardware run
Host: authorized Dell, Chrome headless, Intel UHD Graphics,
vendor `intel`, architecture `gen-12lp`, device `0xa721`.
12 WebGPU compute + rendering steps completed with active GPU.
One GPU step vs independently handwritten CPU gravity+integration:
maximum FP32 state difference 5.960e-8 (allowed 2e-3), zero nonfinite values.
4 focused CPU tests, 345 assertions passed.
No synchronized large-N benchmark, long-run GPU conservation report,
visual screenshot inspection or public Pages deployment is yet claimed.

# Neural Genesis — deterministic GPU morphogenesis slice

This first slice implements real Gray–Scott reaction–diffusion on periodic 2D
fields, not fake animated pigments. It is **not** learned neural cellular
automata or an evolutionary system: replication, mutation, selection,
lineage tracking and learned rules remain unimplemented.

## Numerical model

Two chemical concentrations U and V evolve by a standard Gray–Scott reaction:
`dU/dt = D_u ∇² U - U V² + F(1-U)`,
`dV/dt = D_v ∇² V + U V² - (F+k)V`.

Use a 9-point Laplacian with axial weight 0.2, diagonal weight 0.05,
center coefficient -1. The explicit discrete step uses dt=1 in model
units and concentration clamping [0,1]. Domain wraps periodically.
The finite-difference stencil and nonlinear UV² reaction are handwritten WGSL,
while Handwired emits the final update from a real numeric Circuit.
GPU storage buffers ping-pong without per-frame state readback; render shader
reads the GPU-resident state directly using bilinear display sampling.

## Run and reproduce

```sh
bun install --frozen-lockfile
bun test lab/frontiers/genesis/model.test.ts
bunx tsc -p lab/frontiers/genesis/tsconfig.json --pretty false
bunx oxlint --type-aware lab/frontiers/genesis
bun run site:build
bun scripts/verify-genesis-browser.ts lab/frontiers/genesis/results/local.json --screenshot
```

Hardware verified 2026-10-18 on actual Intel UHD Graphics Gen-12LP
(vendor Intel, device 0xa721) using Chrome's WebGPU. A 192×192 grid
was stepped at least 12 times and compared to an independently calculated CPU
reference after one synchronized step; max absolute concentration difference:
5.960e-8, no nonfinite values. Subsequently captured a real evolved
**1,272-step GPU screenshot** with no CPU state drawing.

Evidence:
- [one-step verification report](results/2026-10-17-evolved-hardware.json)
- [real GPU scene screenshot](results/2026-10-17-evolved-hardware.png)

This is not proof of 100k-cell validated long-term precision, actual
biological morphogenesis, competitive evolution or GPU performance superiority.
Next: long-run CPU/GPU concentration drift and independent spatial
pattern statistics; controlled throughput benchmarks; learned controllers.

# Neural Fluid · Handwired Visual Lab Experiment 04

This independent experiment branch starts directly at GPU-backend commit `daeb82eb5fbd8f3ec4e919b777421de952e9aa9e`.

## Run the actual animation

```sh
bun install --frozen-lockfile
bun test ./lab/visual-fluid/model.test.ts
bun scripts/serve-visual-fluid.ts
```

Open http://127.0.0.1:39085/ in a WebGPU-supported Chrome/Edge. Drag the mouse across either fluid to add momentum and color. The panels show independent CPU and GPU-sourced fluid state. Select solver resolution and WGSL compiler execution strategy.

The `?smoke=1` URL **requires** real WebGPU and evaluates eight frames, comparing full source/output wires on each GPU input batch against the Handwired CPU reference. It writes result JSON into the hidden `#smoke` element and sets `document.body.dataset.status`. No CPU fallback can be marked as an actual GPU pass.

## Exactly what's computed where

| Work | Implementation |
|---|---|
| Backtraced source coordinates / 4-tap gather per cell | CPU host |
| 5-channel bilinear FP32 advection at every cell | **Handwired circuit**, GPU or CPU |
| Mouse dye/momentum injection | CPU host |
| Explicit 5-point velocity/dye diffusion | CPU host |
| 20 Jacobi pressure iterations and velocity projection | CPU host |
| Exterior no-flow boundary enforcement | CPU host |
| Field-color rendering | CPU Canvas2D |

This is a **hybrid semi-Lagrangian fluid solver with a genuine Handwired neural circuit**, not yet fully GPU-resident fluid dynamics. Handwired's 9-point GPU stencil library exists, but the full pressure and fluid solver on GPU is future work. Speedup over a hand-specialized GPU solver is **not claimed**. Frame latency includes gathering, GPU upload, dispatch, readback and host pressure solve; correctness is the first gate.

Inspired by classical stable fluids and the visual style of WebGL fluid demonstrations; algorithm and user interface are independent implementations and do not copy third-party source. No model training.

## Project verification

```sh
bun test ./lab/visual-fluid/model.test.ts
bunx tsc -p .
bunx oxlint --type-aware lab/visual-fluid scripts/serve-visual-fluid.ts scripts/verify-visual-fluid-browser.ts
bun scripts/verify-visual-fluid-browser.ts lab/visual-fluid/results/webgpu.json 128 --screenshot
```

Capture a real hardware adapter name, numeric error, CPU comparison, exit code and screenshots before making public performance claims. The local browser smoke rejects SwiftShader, llvmpipe and all CPU fallbacks.

Next: eliminate per-frame readback using GPU-resident advection + pressure + dye kernels, add divergence and conservation tests, and compare to a handwritten WGSL numerical baseline.

# Handwired Neural Maze · Visual Lab Experiment 02

Branch based directly on general-purpose WebGPU baseline `daeb82eb5fbd8f3ec4e919b777421de952e9aa9e`.

## Run

```sh
bun install --frozen-lockfile
bun test ./lab/visual-maze/model.test.ts
bun scripts/serve-visual-maze.ts
```

Open http://127.0.0.1:39083/. The `?smoke=1` URL runs eight frames in **required real WebGPU mode**, emits `#smoke` JSON, and sets `document.body.dataset.status`. Use the headless smoke verifier (see below) for actual hardware verification.

## Architecture

* Generates a deterministic, solvable maze using iterative recursive-backtracker DFS.
* Performs true level-synchronous breadth-first search by evaluating an identical fixed neural circuit for each cell, in parallel over the GPU batch.
* Input cells include visited, four neighbor frontier bits and passability.
* The circuit ORs frontier bits with ReGLU, gates by obstacles and unseen state, and writes visited/new frontier outputs.
* Host code explicitly gathers neighbors, copies state between iterations and reconstructs the path using BFS distance. These parts are **not** claimed to be GPU-resident or implemented inside the circuit.
* The conventional CPU algorithm is independent of the Handwired circuit. Every hardware sample's full wire table is compared against `circuit.evaluateReference` at tolerance `2e-5`.
* The final path has minimum edge count for a four-connected unweighted maze.

No training or pretrained models.

## Verification

```sh
bunx tsc -p .
bun test ./lab/visual-maze/model.test.ts
bunx oxlint --type-aware lab/visual-maze scripts/serve-visual-maze.ts scripts/verify-visual-maze-browser.ts
bun scripts/verify-visual-maze-browser.ts lab/visual-maze/results/gpu.json --screenshot
```

Project-specific GPU tests must run on actual WebGPU. The smoke harness rejects CPU fallback and software adapters. Record adapter, browser, mismatches, run length and screenshots before public benchmarking. CPU and GPU timings include different host/device overhead and are not kernel-only comparisons.

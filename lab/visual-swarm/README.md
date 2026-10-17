# Handwired Neural Swarm — GPU-resident acceleration

The live Swarm demo has two **different, explicitly labeled** algorithms:

- **Fast · GPU-resident flock (default).** 24×24 spatial-cell occupancy/aggregate grid per species, WebGPU atomic bucket construction, 7×7 neighboring-cell samples, the actual Handwired-generated six-operation steering/integration circuit embedded in WGSL, and instanced triangle/tail drawing directly to a WebGPU canvas. Agent state never leaves GPU memory during normal playback, and the only per-frame CPU→GPU transfer is a 32-byte uniform containing count, pointer, and viewport size. No per-frame JavaScript pairwise neighborhood search, Canvas2D loop over GPU agents, staging/readback or device queue synchronization. This is a **cell-aggregated approximation of Boids**, not exact numerical equivalence to the prior pairwise neighborhood algorithm.
- **Exact · legacy circuit comparison.** CPU reference and GPU evaluate the same 14-source Handwired circuit on identical host-generated pairwise neighborhood inputs, with full readback and wire-by-wire checks. This mode is a correctness diagnostic, not optimized for thousands of agents.

The CPU panel in Fast mode deliberately displays **a separately labeled 256-agent sample** running the legacy exact model. It is not the same-population CPU counterpart of the 16,384-agent GPU model; the two timings must not be used to claim CPU/GPU speedup. The GPU pane displays all selected agents. A full GPU frame's host-side submission time is not the same as actual GPU execution time; the displayed FPS reflects the animation loop cadence.

## Run / test

```sh
bun install --frozen-lockfile
bun test lab/visual-swarm/model.test.ts
bunx tsc -p .
bunx oxlint --type-aware lab/visual-swarm scripts/verify-resident-swarm-browser.ts
bun scripts/serve-visual-swarm.ts
```

Open `http://127.0.0.1:39081/`. Choose 128–16,384 agents, mouse attract/repel, pause/restart, and GPU-resident or legacy mode. For actual hardware validation:

```sh
# GPU compute and instanced rendering, 120 GPU-synchronized simulation frames
bun scripts/verify-resident-swarm-browser.ts lab/visual-swarm/results/resident-16384-120frames.json 16384 --frames=120 --screenshot
# The actual interactive production page in GPU-resident mode
bun scripts/verify-visual-swarm-browser.ts lab/visual-swarm/results/live-resident-16384.json 16384 --resident --screenshot
# Original bit-level CPU-vs-WebGPU source/outputs differential test
bun scripts/verify-visual-swarm-browser.ts lab/visual-swarm/results/live-exact-128.json 128
```

The hardware scripts require real WebGPU (not SwiftShader/CPU fallback). The probe reads GPU positions **once after the timed simulation**, verifies finite/in-bounds state and movement, and records the actual synchronized wall time. This covers compute and draw, not purely GPU kernel timestamps.

## October 17, 2026 — Dell Intel UHD (Gen12LP) measurements

- **New GPU-resident, 16,384 agents:** 120 frames / **537.5 ms** total = **4.479 ms/frame** including GPU completion and instanced rendering; all 16,384 agents moved, none invalid or out of bounds.
- **Old exact pairwise CPU-assisted GPU, 16,384 agents:** example final frame ~**2,049 ms** end-to-end, as measured in the earlier release test; its CPU side took ~1,051.5 ms in that frame.
- **New GPU-resident, 1,024 agents:** 8 synchronized frames / 40.4 ms = **5.05 ms/frame**; 0 invalid agents.
- **Actual interactive 16,384-agent page:** 8 frames passed; 65,536 state scalars checked, 0 invalid/out-of-bounds.
- **Original exact comparison 128 agents:** 8 frames, 20,480 source/output wire checks, zero CPU/GPU mismatches.

The roughly **457× difference in observed per-frame timings** between the old and new 16,384-agent runs is a *whole-implementation comparison with different neighborhood algorithms and run protocols*. It is not a fair identical-algorithm CPU-vs-GPU hardware speedup claim. Repeat measurements under controlled conditions to estimate statistical variance and compare more adapters.

The fast GPU's spatial cell aggregation trades individual-neighbor precision for constant per-agent aggregate samples. It is a meaningful algorithmic optimization, not an exact GPU transcription of the original pairwise solver. For proof of exact original behavior, use legacy mode.

Next research opportunities: exact GPU spatial index and neighbor filtering, GPU timestamp-query profiling, GPU direct Monte Carlo/cell error analysis, CPU-aggregate oracle, and fully isolated adapter benchmarks.

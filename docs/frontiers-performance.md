# Handwired Eight Frontiers — synchronized hardware compute benchmark

Date: 2026-10-18. Captured on authorized Dell using Chrome WebGPU.
GPU adapter: **Intel(R) UHD Graphics**, vendor `intel`,
architecture `gen-12lp`, device `0xa721`.

Source report (machine readable): [Intel UHD six-case benchmark](../lab/frontiers/bench/results/2026-10-17-intel-uhd-optimized.json).

## Measurement contract

Each benchmark uses **identical initial state, numerical parameters and step
count** for its GPU and CPU calculations. The GPU code actually executes the
entire spatial or pairwise numerical algorithm and compiler-generated
Handwired update; no CPU fallback is accepted by the verifier.
The GPU interval includes issuing compute passes and an
**awaited queue completion fence**, not just host dispatch submission.
One final GPU staging-buffer readback follows and is separately measured.

Shader compilation and device creation are measured separately and excluded
from the GPU steady-state computation column. Rendering, cold startup, initial
uploads and transfer are not included in this physics-computation ratio.
Each solver is warmed up and reset before the timed GPU batch.
The CPU includes the equivalent numerical steps; the data-oriented Genesis
CPU includes construction of its precomputed periodic-neighbor table.
The CPU oracle is independently handwritten, not a GPU or Handwired fallback.

These are **one-trial host-observed timings**, not GPU timestamp-query kernel
timings. Variation across loads, browser scheduling, thermal states and GPU
adapters has not been characterized. Ratios compare implemented algorithms
and software runtimes, not intrinsic Intel hardware throughput.

## Actual results

| Solver | Workload | Steps | Strong CPU ms | Synchronized GPU ms | GPU readback ms | CPU/GPU ratio | Max absolute CPU/GPU error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Universe pairwise | 128 bodies | 40 | 14.7 | 7.2 | 6.0 | 2.04× | 1.79e-7 |
| Universe pairwise | 256 bodies | 40 | 36.9 | 10.0 | 5.0 | 3.69× | 2.51e-7 |
| Universe pairwise | 512 bodies | 40 | 103.2 | 15.2 | 3.6 | 6.79× | 5.36e-7 |
| Genesis 9-point | 96² cells | 40 | 33.9 | 4.6 | 4.6 | 7.37× | 2.80e-6 |
| Genesis 9-point | 192² cells | 40 | 41.5 | 4.9 | 3.9 | 8.47× | 9.18e-6 |
| Genesis 9-point | 384² cells | 12 | 75.1 | 7.2 | 5.6 | 10.43× | 5.36e-7 |

All six hardware comparisons completed with **zero nonfinite outputs**.
Numerical differences are maximum absolute differences after the specified
number of steps, not bitwise equality claims.

The Genesis scalar reference took 246.5, 527.8 and 473.4 ms, respectively,
but was deliberately **not used as the primary comparison**. A faster
typed-array CPU stencil with cached periodic neighbors was implemented
independently and its outputs matched the scalar oracle *exactly* in the
tested cases. The optimized CPU was the stronger measured benchmark.

Universe energy relative drift after 40 steps on the GPU was approximately
3.13e-4 (128 bodies), 3.55e-5 (256 bodies), and 4.77e-4 (512 bodies).
Maximum momentum errors were on the order of 1e-8 in model units. These
short trajectories do not establish bounded drift over long encounters.

## Reproduction

```sh
bun install --frozen-lockfile
bun test lab/frontiers/bench/reference-fast.test.ts
bunx tsc -p lab/frontiers/bench/tsconfig.json
bun scripts/benchmark-frontiers-browser.ts lab/frontiers/bench/results/local-run.json
```

The last command uses an actual Chromium WebGPU adapter, rejects known
software adapters, records CPU, GPU, setup, staging-readback and errors, and
safely terminates its ephemeral browser profile. It should not be run as a
substitute for hardware verification on unrelated adapters.

## Remaining measurement work

1. Repeat 10–30 controlled trials and show medians, percentiles and uncertainty.
2. Add optional timestamp-query kernel timings with explicit feature detection.
3. Separate render, transfer and presentation frame time in the interactive apps.
4. Compare stronger JS/WASM/SIMD CPU solvers where appropriate and disclose methodology.
5. Cross-check other physical GPU adapters, including dedicated cards when browser
   driver access is available. Kaggle CUDA access alone is **not** Chromium WebGPU proof.
6. Scale algorithms scientifically (Barnes–Hut and hierarchical spatial indexing);
   no million-body hardware acceleration or simulation accuracy is currently claimed.

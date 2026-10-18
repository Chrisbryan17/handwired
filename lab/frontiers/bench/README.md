# Frontiers synchronized benchmark fixture
This is deliberately a **private-to-the-build verification fixture**, not a public
benchmark gallery. It measures strictly identified real WebGPU adapters.

- CPUs: handwritten `stepUniverseCpu` and `stepGenesisCpu` oracles;
- GPUs: exact same initial conditions, parameters, state dimensions and step counts;
- Warm-up: 3 GPU steps, synchronization, re-upload original state;
- GPU compute metric: host wall time from command submission through
  `queue.onSubmittedWorkDone()`. Not GPU timestamp-query kernel time.
- GPU readback: separately measured `read()`, excluded from computation time.
- Setup: device and shader compilation separately tracked and excluded.
- Sampling: one trial per workload; no universal speedup is justified.
- Numerical quality: max absolute error, RMSE, finite checks, and orbital
  energy/momentum drift measured from actual readback.

Universe: 128, 256, 512 bodies × 40 steps; O(N²) all-pairs physics.
Genesis: 96²,192² cells × 40 steps and 384² × 12 steps.
Running local benchmark requires real Chrome/Edge WebGPU and may load the
GPU; don't run it in CI where software GPU fallbacks are common.

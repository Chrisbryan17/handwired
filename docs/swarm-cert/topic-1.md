# WebGPU Reliability and Verification

The pinned implementation makes CPU evaluation the reference point and exposes hardware FP32 as a distinct policy. Recommendations below follow the checked-in source excerpts; README performance and differentiability claims do not establish runtime correctness.

## Reliability

- Treat `reference-exact` and `hardware-f32` as different contracts. In `src/runtime/webgpu.ts`, `create` rejects GPU execution when exact semantics are requested, and `auto` selects CPU for that policy. Callers that require reference behavior should request it explicitly and check `mode` and `fallbackReason`; GPU success alone does not mean CPU-identical arithmetic.
- The shader in `src/runtime/webgpu-shader.ts` rounds operation terms through FP32 arithmetic, but driver contraction or subnormal handling may still differ. Its boolean operation uses a `>= 0.5` threshold documented as equivalent only for inputs in `[0,1]`; test values near the boundary and verify that callers uphold that input domain. Index reads also have distinct NaN, below-range, and above-range paths, so exercise each policy and validate candidate tables.
- GPU execution retains mutable source and value buffers. The backend documentation says methods must not run concurrently, and device loss invalidates GPU-only state. Serialize calls per evaluator and retain or reconstruct a known source checkpoint before recreating an evaluator after loss.
- `auto` falls back on device initialization failure, while `gpu` propagates failure. Surface `fallbackReason` in diagnostics so a successful CPU result is not mistaken for a GPU run. Capacity and storage binding limits can also cause fallback or explicit failure.

## Verification

Use `evaluateReference` as the oracle and compare every output wire, not only final application results. Cover mixed operations, repeated state updates, omitted source updates, empty-source batches, boundary and nonfinite values, and capacity rejection. Run tests on actual browser adapters: the docs explicitly note that observed matches on one adapter do not guarantee bit identity across hardware. Record adapter/browser versions and numerical policy with each result.

The existing browser verification commands in `docs/webgpu-backend.md` include a mixed-circuit check, state updates, and device destruction. Extend that evidence with repeated fault and resource-limit runs, and report fallback outcomes separately from GPU outcomes. The repository suite also has a noted legacy parsing failure, so report that baseline limitation instead of claiming the full suite is clean.

## Team handoff

Please use these checks as release gates for reliability claims: publish per-adapter differential results and boundary coverage, make fallback state visible in benchmark output, and track device-loss recovery and resource-exhaustion tests as open work. Keep numerical-policy semantics and supported boolean input ranges explicit in API-facing documentation.
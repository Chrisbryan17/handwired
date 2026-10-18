# Handwired Frontiers — reusable WebGPU 3D spatial grid

## Implemented component

A concrete GPU-resident linked-cell **uniform spatial index** for local
interactions, built as a shared Handwired runtime library rather than being
hardcoded to fluid, cloth, agents, or robotics.

- `src/runtime/frontiers-spatial.ts`: strict geometry/grid validation,
  deterministic 3D data scenarios and **independent all-pairs CPU oracle**.
- `src/runtime/frontiers-spatial-gpu.ts`: GPU atomic cell-head reset,
  concurrent linked-list scatter and 27-cell (current plus 26 adjacent)
  Euclidean radius query. Each particle gets its **actual neighbor count**;
  it is not an occupancy-colored visualization or a CPU fallback.
- `lab/frontiers/spatial/probe.ts` and `scripts/verify-spatial-browser.ts`:
  actual Chrome WebGPU compute, adapter identification, synchronized queue
  completion and explicit result readback against CPU reference.

## Geometry contract

Positions are packed `vec4<f32>` (xyz + user payload); axis-aligned
world bounds use 3D origin, cell dimensions and positive cell size.
Radius must be **≤ cell size**, so 27 adjacent cells cover all possible
true neighbors. Each distinct other point at Euclidean separation
`distance² ≤ radius²` counts; a point does not count itself.
Points outside the domain are excluded completely.
Cell indices use `(z * dimY + y) * dimX + x`.

This is a **linked-cell** index, not a sorted/hierarchical hash, GPU radix
sort, BVH or a collision response solver. Atomic insertion makes
traversal order nondeterministic, but exact neighbor counts remain
deterministic. The kernel bounds list traversals to prevent infinite loops.

## WebGPU execution and resource management

The index uses 5 persistent WebGPU allocations: input position storage,
atomic heads, linked-next, output neighbor counts, and a packed uniform.
One command encoder dispatches cell reset, atomic insertion and query.
No CPU state readback is performed within `encodeQuery`, `queryNow`
or when consumer render passes use the resident buffers.
`readCounts()` is an explicit synchronized testing/diagnostic step.

The host checks all dimensions, limits and finite inputs, rejects known
software GPU adapters, and destroys the tracked resources during disposal.
It is legal to update positions with `uploadPoints()`, but an unchanged
point count is required within a solver instance.

## Hardware-corroborated correctness

Intel UHD Graphics, vendor `intel`, Gen-12LP, device `0xa721`
via actual Chromium WebGPU on 2026-10-18.

- **384 scattered points**, 1,000 spatial cells: 268 directed neighbor
  relationships, 0 GPU/CPU mismatches.
- **128 clustered points**, one outside the domain: 16,002 directed
  neighbor relationships, 0 GPU/CPU mismatches.
- Both runs identify the real adapter and explicitly read GPU counts after
  query completion. Machine-readable evidence and a screenshot:
  [results/2026-10-18-intel-uhd.json](../../lab/frontiers/spatial/results/2026-10-18-intel-uhd.json).

GPU query timings in the report are **single-trial synchronized host times**,
not isolated GPU timestamp-query kernel durations or speedup claims.

## Tests

```sh
bun test src/runtime/frontiers-spatial.test.ts
bunx tsc -p .
bunx tsc -p lab/frontiers/spatial/tsconfig.json
bunx oxlint --type-aware src/runtime/frontiers-spatial.ts src/runtime/frontiers-spatial-gpu.ts lab/frontiers/spatial
bun scripts/verify-spatial-browser.ts lab/frontiers/spatial/results/local.json
```

## Next shared kernels

1. Prefix-summed uniform grid / radix-scattered contiguous neighbors for
   more predictable memory locality and throughput.
2. Neighbor iteration APIs that apply a callback / generated numerical
   Handwired kernel inside the GPU pass, not count-only outputs.
3. Periodic and reflective boundaries; variable-radius and multilevel grids.
4. GPU radix sort, BVH/AABB construction and narrow-phase collision data
   consumed by Destruction, Cataclysm, Earth and Machines.
5. Repeated large-scene performance benchmarks and cross-adapter validation.

Correctness of a linked-cell query is not proof of a complete cloth solver,
fluid simulation, multiagent controller or rigid-body collision system.

# Neural Light — first working GPU path-tracing research slice

Handwired's Neural Light implements a real WebGPU Monte Carlo path tracer,
progressively accumulating radiance in a GPU storage buffer. It is not
pre-rendered animation, a rasterization-only demo or a model viewer.

## What runs on GPU

- Rays intersect three analytic spheres and five bounded room planes, plus
  a rectangular emissive area light.
- Lambertian cosine-weighted diffuse scattering, next-event area-light
  sampling with shadow rays, perfect mirror reflection, dielectric refraction
  with Schlick Fresnel, Russian roulette and multiple ray bounces.
- The RGB path throughput product is compiled from a **real Handwired Circuit**,
  embedded in the WGSL generated compute kernel.
- Each image frame contributes one stochastic sample per pixel to a
  **resident, persistent** RGB+sample-count accumulation buffer.
- Filmic exposure and gamma output in a separate GPU render pass.
- Real camera orbit, sample reset, emitter intensity, refractive-index,
  depth, exposure, resolution, desktop/mobile interface and PNG export.

The scene is deterministic synthetic geometry and does **not** contain
external imagery or textures.

## Actual verification

Intel UHD Graphics Gen-12LP (`intel / gen-12lp / 0xa721`) completed
12-spp WGSL multi-bounce path tracing on 19,200 pixels with no
nonfinite values, with independent CPU reference smoke. A later
real desktop run rendered 200 samples per pixel at 240x180, with
a mobile 59-spp screen capture. See the unedited hardware reports and PNGs
under [results](results/).

The `geometry-verify.ts` fixture independently compares GPU sphere,
room plane, emitter, normal orientation and miss traversal against
the CPU `model.ts` ray oracle. These are deterministic finite-precision
geometry tests, **not** evidence of pixel-perfect stochastic transport.
The CPU source contains separately tested Lambertian and specular
transport paths.

## Run

```sh
bun install --frozen-lockfile
bun test lab/frontiers/light/model.test.ts
bunx tsc -p lab/frontiers/light/tsconfig.json --pretty false
bunx oxlint --type-aware lab/frontiers/light
bun run site:build
bun scripts/verify-light-browser.ts lab/frontiers/light/results/local-hardware.json --screenshot
bun scripts/verify-light-demo-browser.ts lab/frontiers/light/results/local-desktop.json --samples=96 --screenshot
```

Each hardware probe requires an actual GPU adapter. A software GPU
is rejected, and the temporary test browser is cleaned up by profile.
The route to the public demo is `/handwired/neural-light/` once Pages
deployment succeeds.

## Deliberately open gates

BVH acceleration and generic meshes; GGX/PBR materials; MIS and unbiased
radiance estimator checks; statistical CPU/GPU convergence over many random
seeds; GPU timestamp timings, equal-quality CPU performance baselines;
dynamic scene editing; denoising; pixel-scale convergence controls; GPU
resource/loss stress and cross-adapter reproducibility. This first slice
must not be advertised as a completed photorealistic renderer.

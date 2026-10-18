# Neural Light — path tracing vertical-slice ledger

Current stage: CPU reference implementation. Hardware compute, preview, BVH,
material editing, GPU ray tracing and progress accumulation are upcoming.

The CPU oracle includes exact sphere and bounded plane intersections,
a synthetic Cornell test room, area-light next-event estimation,
cosine-weighted hemisphere sampling, mirror reflection, dielectric
refraction with Schlick Fresnel, multi-bounce path transport,
Russian roulette and reproducible pseudorandom sampling.

A real Handwired `Circuit` computes three-channel material-throughput
multiplication, validated against an independent handwritten reference.
Source geometry is wholly synthetic and MIT original code; it is **not**
a reconstructed real scene or a pretended trained lighting model.

Scientific scope: diffuse Lambert plus idealized delta specular/dielectric.
No BVH, participating media, MIS, spectral rendering, denoising or hardware
claims at this CPU milestone.

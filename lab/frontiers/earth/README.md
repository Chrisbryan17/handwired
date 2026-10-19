# Neural Earth: authentic Kingston OSM routing and traffic foundation

This is a numerical graph/traffic vertical slice, **not** a complete
city simulation and **not yet GPU verified**. The only accepted world
coordinates are actual WGS84/OpenStreetMap data loaded by the caller.
Synthetic routes and vehicle schedules in tests are labeled fixtures,
not fabricated real-world geodata.

`model.ts` parses OpenStreetMap Overpass elements, builds directionally
correct road segments, handles mapped legal one-way restrictions and
maxspeed attributes, computes meter-scale Haversine distances, and
runs travel-time shortest paths accounting for closures, congestion
occupancy and signal green fractions. Vehicles advance through real
segment lengths; edge/path conservation and deterministic CPU tests
verify consistency.

Next independent gates: acquire and retain the actual licensed
Kingston OSM snapshot with exact source timestamp, attribution
(© OpenStreetMap contributors, ODbL), traffic light junction mapping,
GIS/rendering, authentic city geometry, GPU-resident large populations,
emergency/evacuation and a real 3D browser preview.

Do **not** bulk scrape OSM public raster/vector tiles. Follow their
published tile and ODbL attribution rules. This code only parses OSM
graph data and has no tile-server dependency.

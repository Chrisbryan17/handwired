import {expect,test} from 'bun:test'
import {parseOsmRoads,fastestRoute,haversine,edgeTravelSeconds,
 edgeKey,advanceVehiclesCpu,localMeters,type OsmData} from './model.ts'
const osm:OsmData={elements:[
 {type:'node',id:1,lat:18.014,lon:-76.793},
 {type:'node',id:2,lat:18.015,lon:-76.793},
 {type:'node',id:3,lat:18.016,lon:-76.793},
 {type:'node',id:4,lat:18.015,lon:-76.792},
 {type:'way',id:11,nodes:[1,2,3],tags:{highway:'residential'}},
 {type:'way',id:12,nodes:[2,4,3],tags:{highway:'primary',oneway:'yes',maxspeed:'50'}},
]}
test('WGS84 haversine and local Kingston meters are physical scales',()=>{
 expect(haversine([18.014,-76.793],[18.015,-76.793])).toBeGreaterThan(110)
 expect(localMeters(18.014,-76.793)[0]).toBe(0)
 expect(localMeters(18.015,-76.793)[1]).toBeCloseTo(111.195,1)
})
test('OSM ways create correct graph, preserve one-way restriction and assigned real lengths',()=>{
 const g=parseOsmRoads(osm)
 expect(g.adjacency.get(4)?.some(e=>e.to===3)).toBe(true)
 expect(g.adjacency.get(3)?.some(e=>e.to===4)).toBe(false)
 expect(g.adjacency.get(2)?.find(e=>e.to===4)?.maxMetersPerSecond).toBeCloseTo(50/3.6)
})
test('Dijkstra travel time routes obey topology, closures and independent cost oracle',()=>{
 const graph=parseOsmRoads(osm)
 const route=fastestRoute(graph,1,3)
 expect(route?.nodes).toEqual([1,2,3])
 const slowEdge=graph.adjacency.get(2)!.find(e=>e.to===3)!
 const state={closures:new Set([edgeKey(slowEdge)])}
 const diverted=fastestRoute(graph,1,3,state)
 expect(diverted?.nodes).toEqual([1,2,4,3])
 const independent=diverted!.nodes.slice(1).reduce((acc,to,i)=>{
  const e=graph.adjacency.get(diverted!.nodes[i]!)!.find(edge=>edge.to===to)!
  return acc+edgeTravelSeconds(e,state)
 },0)
 expect(diverted?.travelSeconds).toBeCloseTo(independent,9)
})
test('crowding and signal timing change actual route costs',()=>{
 const graph=parseOsmRoads(osm)
 const baseline=fastestRoute(graph,1,3)!
 const congested=fastestRoute(graph,1,3,{
  occupancy:new Map([['1:2',.95]]),
  signals:new Map([[2,{greenFraction:.2}]])
 })
 expect(congested?.travelSeconds).toBeGreaterThan(baseline.travelSeconds)
})
test('vehicle movement advances along real segments, not animated interpolation',()=>{
 const g=parseOsmRoads(osm),r=fastestRoute(g,1,3)!
 const initial={id:9,route:r.nodes,segment:0,progressMeters:0,velocity:10}
 const after=advanceVehiclesCpu([initial],g,.25)
 expect(after[0]?.progressMeters).toBeCloseTo(2.5,8)
 expect(after[0]?.segment).toBe(0)
 const repeated=Array.from({length:60}).reduce<ReturnType<typeof advanceVehiclesCpu>>(
  v=>advanceVehiclesCpu(v,g,.25),[initial])
 expect(repeated[0]!.segment).toBeGreaterThanOrEqual(1)
})
test('bad OSM and negative occupancies fail rather than silently fabricate roads',()=>{
 expect(()=>parseOsmRoads({elements:[]})).toThrow()
 const g=parseOsmRoads(osm)
 expect(()=>fastestRoute(g,1,3,{occupancy:new Map([['1:2',-1]])})).toThrow()
})

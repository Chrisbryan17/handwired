/** Kingston OpenStreetMap road network numerical reference.
 * Consumes real OSM/Overpass features. No generated roads or imagery.
 * Traffic volumes and signal timings in simulations are explicitly synthetic.
 */
export type OsmNode={type:'node',id:number,lat:number,lon:number}
export type OsmWay={type:'way',id:number,nodes:number[],tags?:Record<string,string>}
export type OsmData={elements:(OsmNode|OsmWay)[]}
export type RoadEdge={from:number,to:number,meters:number,maxMetersPerSecond:number,
 roadType:string,osmWay:number}
export type KingstonRoadGraph={nodes:Map<number,[number,number]>,
 adjacency:Map<number,RoadEdge[]>}
export const kingstonOrigin={lat:18.014,lon:-76.793}
const earthRadius=6371008.8
export function localMeters(lat:number,lon:number,
 center=kingstonOrigin):readonly [number,number]{
 if(![lat,lon,center.lat,center.lon].every(Number.isFinite)||Math.abs(lat)>90||
  Math.abs(lon)>180)throw Error('invalid WGS84 coordinates')
 return [earthRadius*(lon-center.lon)*Math.PI/180*Math.cos(center.lat*Math.PI/180),
  earthRadius*(lat-center.lat)*Math.PI/180]
}
export function haversine(a:readonly [number,number],
 b:readonly [number,number]):number{
 const p=Math.PI/180,lat1=a[0]*p,lat2=b[0]*p
 const dLat=(b[0]-a[0])*p,dLon=(b[1]-a[1])*p
 const v=Math.sin(dLat/2)**2+
  Math.cos(lat1)*Math.cos(lat2)*Math.sin(dLon/2)**2
 return 2*earthRadius*Math.asin(Math.min(1,Math.sqrt(v)))
}
const defaultSpeed:Record<string,number>={
 motorway:22,trunk:20,primary:14,secondary:12,tertiary:11,
 residential:8,unclassified:8,living_street:4,service:5
}
export function parseOsmRoads(data:OsmData):KingstonRoadGraph{
 if(!Array.isArray(data.elements)||data.elements.length>150_000)
  throw Error('OSM road dataset invalid or too large')
 const nodes=new Map<number,[number,number]>()
 const adjacency=new Map<number,RoadEdge[]>()
 for(const x of data.elements)if(x.type==='node'){
  if(!Number.isSafeInteger(x.id)||!Number.isFinite(x.lat)||!Number.isFinite(x.lon)||
    x.lat < -90||x.lat>90||x.lon < -180||x.lon>180)
   throw Error('invalid OSM node geometry')
  nodes.set(x.id,[x.lat,x.lon])
 }
 let roadCount=0
 const add=(e:RoadEdge)=>{const list=adjacency.get(e.from)??[];list.push(e);adjacency.set(e.from,list)}
 for(const x of data.elements)if(x.type==='way'){
  const kind=x.tags?.['highway']
  if(kind===undefined||!(kind in defaultSpeed)||x.nodes.length<2)continue
  if(!Number.isSafeInteger(x.id))throw Error('invalid OSM road way id')
  let speed=defaultSpeed[kind]!
  const maxspeed=x.tags?.['maxspeed']
  if(maxspeed!==undefined){
   const match=/^(\d{1,3})(?:\s*km\/h)?$/.exec(maxspeed.trim())
   if(match)speed=Math.min(140,Math.max(2,Number(match[1])))/3.6
  }
  const direction=x.tags?.['oneway']==='-1'?-1:
   x.tags?.['oneway']==='yes'||x.tags?.['junction']==='roundabout'?1:0
  for(let i=1;i<x.nodes.length;i++){
   const a=x.nodes[i-1]!,b=x.nodes[i]!,pa=nodes.get(a),pb=nodes.get(b)
   if(pa===undefined||pb===undefined||a===b)continue
   const meters=haversine(pa,pb)
   if(meters<=0)continue
   if(!Number.isFinite(meters)||meters>10000)throw Error('invalid OSM road segment length')
   const edge={from:a,to:b,meters,maxMetersPerSecond:speed,roadType:kind,osmWay:x.id}
   if(direction>=0)add(edge)
   if(direction<=0)add({...edge,from:b,to:a})
   roadCount++
  }
 }
 if(roadCount===0)throw Error('OSM dataset contains no routable road segments')
 return {nodes,adjacency}
}
export type RoadState={closures?:Set<string>,occupancy?:Map<string,number>,
 signals?:Map<number,{greenFraction:number}>}
export const edgeKey=(edge:Pick<RoadEdge,'from'|'to'>)=>edge.from+':'+edge.to
export function edgeTravelSeconds(e:RoadEdge,state:RoadState={}):number{
 const traffic=state.occupancy?.get(edgeKey(e))??0
 if(!Number.isFinite(traffic)||traffic<0||traffic>1)
  throw Error('invalid traffic occupancy')
 if(state.closures?.has(edgeKey(e)))return Infinity
 const signal=state.signals?.get(e.to)?.greenFraction??1
 if(!Number.isFinite(signal)||signal<=0||signal>1)
  throw Error('invalid signal green fraction')
 const velocity=e.maxMetersPerSecond*Math.max(.08,1-traffic*.92)
 return e.meters/velocity+(1-signal)*24
}
export type RouteResult={nodes:number[],distanceMeters:number,travelSeconds:number,
 expanded:number}
export function fastestRoute(graph:KingstonRoadGraph,from:number,to:number,
 state:RoadState={}):RouteResult|null{
 if(!graph.nodes.has(from)||!graph.nodes.has(to))throw Error('unknown OSM endpoints')
 const costs=new Map([[from,0]]),distance=new Map([[from,0]]),
 predecessors=new Map<number,number>(),done=new Set<number>()
 const frontier:{node:number,cost:number}[]=[{node:from,cost:0}]
 let expanded=0
 while(frontier.length){
  frontier.sort((a,b)=>a.cost-b.cost)
  const u=frontier.shift()!
  if(done.has(u.node))continue
  done.add(u.node);expanded++
  if(u.node===to)break
  for(const e of graph.adjacency.get(u.node)??[]){
   const weight=edgeTravelSeconds(e,state)
   if(!Number.isFinite(weight)||done.has(e.to))continue
   const alt=u.cost+weight
   if(alt<(costs.get(e.to)??Infinity)){
    costs.set(e.to,alt)
    distance.set(e.to,(distance.get(u.node)??0)+e.meters)
    predecessors.set(e.to,u.node)
    frontier.push({node:e.to,cost:alt})
   }
  }
 }
 if(!costs.has(to))return null
 const path=[to]
 while(path[0]!==from){
  const predecessor=predecessors.get(path[0]!)
  if(predecessor===undefined)throw Error('inconsistent reconstructed route')
  path.unshift(predecessor)
 }
 return {nodes:path,distanceMeters:distance.get(to)!,travelSeconds:costs.get(to)!,expanded}
}
export type Vehicle={route:number[],segment:number,progressMeters:number,velocity:number,
 id:number}
export function advanceVehiclesCpu(vehicles:readonly Vehicle[],graph:KingstonRoadGraph,
 seconds:number,state:RoadState={}):Vehicle[]{
 if(!Number.isFinite(seconds)||seconds<0||seconds>.25)throw Error('invalid traffic integrator dt')
 return vehicles.map(v=>{
  if(v.segment>=v.route.length-1)return {...v}
  let distance=Math.max(0,v.velocity*seconds),segment=v.segment,progress=v.progressMeters
  if(!Number.isFinite(distance)||!Number.isFinite(progress)||progress<0)
   throw Error('invalid vehicle state')
  for(let hops=0;hops<6&&segment<v.route.length-1;hops++){
   const edge=graph.adjacency.get(v.route[segment]!)?.find(e=>e.to===v.route[segment+1])
   if(!edge)throw Error('vehicle route contains non-road connection')
   if(!Number.isFinite(edgeTravelSeconds(edge,state)))break
   const remaining=Math.max(0,edge.meters-progress)
   if(distance<remaining){progress+=distance;distance=0;break}
   distance-=remaining;progress=0;segment++
   if(distance===0)break
  }
  return {...v,segment,progressMeters:progress}
 })
}

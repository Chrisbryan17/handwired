import type {SphereBodies, SphereContactSettings} from '../../../src/runtime/frontiers-contacts.ts'
import type {SpatialGridConfig} from '../../../src/runtime/frontiers-spatial.ts'

/** Geometry is synthetic; bodies are simulated, not pre-scripted animation. */
export type ImpactPreset='impact'|'cascade'|'swarm'
export const impactSpatialGrid:SpatialGridConfig={
 origin:[-13,-11,-13],dimensions:[18,18,18],cellSize:1.5,radius:1.35,
}
export const impactDefaults:SphereContactSettings={
 dt:1/60,gravity:[0,-9.81,0],restitution:.35,friction:.5,
 penetrationSlop:.0008,correctionFraction:.75,iterations:1,
}
export function makeImpactScenario(preset:ImpactPreset):SphereBodies{
 const positions:number[]=[],velocities:number[]=[]
 const add=(x:number,y:number,z:number,radius:number,
  vx:number,vy:number,vz:number,inverseMass=1)=>{
  positions.push(Math.fround(x),Math.fround(y),Math.fround(z),Math.fround(radius))
  velocities.push(Math.fround(vx),Math.fround(vy),Math.fround(vz),Math.fround(inverseMass))
 }
 if(preset!=='swarm'){
  // Independently stationary spheres emulate a rough collision floor.
  // This is NOT a watertight infinite plane or a fracture model.
  for(let x=-4;x<=4;x++)for(let z=-4;z<=4;z++)
   add(x*1.10,-2.45,z*1.10,.62,0,0,0,0)
 }
 if(preset==='impact'){
  for(let layer=0;layer<5;layer++){
   for(let x=0;x<5;x++){
    for(let z=0;z<2;z++){
     add((x-2)*.86,-1.15+layer*.85,(z-.5)*.86,.44,0,0,0)
    }
   }
  }
  add(-7.6,-.2,0,.64,13.8,0,0,.35)
 }else if(preset==='cascade'){
  for(let layer=0;layer<6;layer++){
   for(let x=0;x<5;x++)for(let z=0;z<3;z++){
    add((x-2)*.84,-1.12+layer*.84,(z-1)*.87,.43,
      0,0,0,1)
   }
  }
  add(0,7.4,0,.67,.15,-8,0,.26)
 }else {
  // Seeded, repeatable multiagent collision swarm (no learned policy).
  let seed=854633
  const next=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}
  for(let i=0;i<96;i++){
   const a=next()*Math.PI*2
   const b=(next()-.5)*Math.PI
   const r=2.3+next()*4
   const x=r*Math.cos(a)*Math.cos(b),y=r*Math.sin(b),z=r*Math.sin(a)*Math.cos(b)
   const inv=(i%13===0)?0:1
   add(x,y,z,.35+next()*.12,-x*.42,-y*.42,-z*.42,inv)
  }
 }
 return {positions:Float32Array.from(positions),
  velocities:Float32Array.from(velocities)}
}

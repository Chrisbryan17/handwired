import {SphereContactGpu} from '../../../src/runtime/frontiers-contacts-gpu.ts'
import {stepSphereContactsCpu,type SphereBodies,type SphereContactSettings} from '../../../src/runtime/frontiers-contacts.ts'
import {seededSpatialPoints,type SpatialGridConfig} from '../../../src/runtime/frontiers-spatial.ts'

const settings:SphereContactSettings={dt:.012,gravity:[0,-9.81,0],
  restitution:.55,friction:.32,correctionFraction:.75,
  penetrationSlop:.0005,iterations:1}
const grid:SpatialGridConfig={origin:[-5,-5,-5],
  dimensions:[16,16,16],cellSize:1,radius:1}
const couple:SphereBodies={
 positions:Float32Array.of(
  -1,0,0,.4,-.27,0,0,.4,2,1,0,.5,2.87,1,0,.5,
  .4,3,0,.3,.4,4,0,.3),
 velocities:Float32Array.of(
  1,.1,0,1,-.4,-.1,0,1,.4,0,0,1,-.1,0,0,0,
  0,-.3,0,1,0,0,0,1),
}
const scattered:SphereBodies={positions:seededSpatialPoints(64,39),
 velocities:new Float32Array(64*4)}
for(let i=0;i<64;i++){
 scattered.positions[i*4+3]=.22
 scattered.velocities[i*4]=Math.fround((i%7-3)*.09)
 scattered.velocities[i*4+1]=Math.fround((i%9-4)*.07)
 scattered.velocities[i*4+2]=Math.fround((i%5-2)*.08)
 scattered.velocities[i*4+3]=1
}
const scenarios=[{name:'multi-contact-plus-anchor',body:couple},
 {name:'64-body-gravity-and-sparse-contacts',body:scattered}]
const status=document.getElementById('result')
const results=[]
let adapter=''
try{
 for(const scenario of scenarios){
  const cpu=stepSphereContactsCpu(scenario.body,settings)
  const gpu=await SphereContactGpu.create(scenario.body,grid,settings)
  try{
   adapter=gpu.grid.gpu.adapter.name
   const start=performance.now()
   gpu.step();await gpu.flush()
   const synchronizedMs=performance.now()-start
   const readback=performance.now()
   const actual=await gpu.readState()
   const readbackMs=performance.now()-readback
   let invalid=0,maxError=0,meanError=0
   for(const field of ['positions','velocities'] as const){
    const reference=cpu.state[field],observed=actual[field]
    for(let i=0;i<reference.length;i++){
     const value=observed[i]??Number.NaN
     const target=reference[i]??Number.NaN
     if(!Number.isFinite(value)||!Number.isFinite(target))invalid++
     const error=Math.abs(value-target)
     maxError=Math.max(maxError,error)
     meanError+=error
    }
   }
   results.push({scenario:scenario.name,bodies:scenario.body.positions.length/4,
     steps:gpu.steps,contactsProcessedByCpu:cpu.contactsProcessed,
     synchronizedMs,readbackMs,maxError,invalid,
     meanError:meanError/(actual.positions.length+actual.velocities.length),
     tolerance:.002,passes:invalid===0&&maxError<=.002,
     trackedGpuBytes:gpu.grid.gpu.bytesAllocated()})
  }finally{gpu.dispose()}
 }
 if(status)status.textContent=JSON.stringify({ok:results.every(x=>x.passes),
   adapter,algorithm:'GPU spatial broadphase + prediction + Jacobi sphere contacts',
   oracle:'independently handwritten CPU impulses, restitution, friction, correction',
   results,physicalGpuExpected:true,
   scope:'one contact solver step, not a full rigid body or robotics simulator'})
}catch(error){
 if(status)status.textContent=JSON.stringify({ok:false,error:String(error)})
}

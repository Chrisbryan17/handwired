import {BondedStructureGpu} from './fracture-gpu.ts'
import {makeBondedBridge,stepBondedCpu,applyStructuralImpulse,
  type BondedBody,type BondedSettings} from './fracture-model.ts'
const result=document.getElementById('result')
const settings:BondedSettings={dt:.004,gravity:[0,-9.81,0]}
function tensionBreak():BondedBody{
 const initial=makeBondedBridge(6,3,.65)
 const copy:BondedBody={...initial,positions:initial.positions.slice()}
 // Geometric overstretch is the actual reason for fracture, not a
 // scripted fixed-time collapse event.
 const node=5*3*2+1*2+0
 copy.positions[node*4]=Math.fround(copy.positions[node*4]!+2.2)
 return copy
}
const scenarios=[
 {name:'elastic-loaded-bridge',input:applyStructuralImpulse(makeBondedBridge(6,3),19,[.2,-6,.5]),steps:1},
 {name:'strain-triggered-fracture',input:tensionBreak(),steps:1},
 {name:'ten-step-gravity-and-impulse',input:applyStructuralImpulse(makeBondedBridge(6,3),15,[2,-2,.1]),steps:10},
]
const results=[]
let adapter=''
try{
 for(const scenario of scenarios){
  let oracle=scenario.input,events=0
  for(let step=0;step<scenario.steps;step++){
   const actual=stepBondedCpu(oracle,settings)
   oracle=actual.body;events+=actual.diagnostics.events
  }
  const gpu=await BondedStructureGpu.create(scenario.input,settings)
  try{
   adapter=gpu.gpu.adapter.name
   const start=performance.now()
   for(let step=0;step<scenario.steps;step++)gpu.step()
   await gpu.flush()
   const completedMs=performance.now()-start
   const reading=performance.now(),actual=await gpu.read()
   const readbackMs=performance.now()-reading
   let invalid=0,maxError=0,damageError=0,activeMismatches=0
   for(const key of ['positions','velocities'] as const){
    for(let i=0;i<actual[key].length;i++){
     const v=actual[key][i]!
     if(!Number.isFinite(v))invalid++
     maxError=Math.max(maxError,Math.abs(v-oracle[key][i]!))
    }
   }
   for(let i=0;i<oracle.endpoints.length/2;i++){
    const base=i*8
    const damage=actual.bonds[base+6]!
    if(!Number.isFinite(damage))invalid++
    damageError=Math.max(damageError,Math.abs(damage-oracle.bonds[base+6]!))
    if(actual.bonds[base+7]!==oracle.bonds[base+7])activeMismatches++
   }
   const broken=Array.from(actual.bonds).filter((_,i)=>i%8===7&&actual.bonds[i]===0).length
   results.push({name:scenario.name,nodes:gpu.nodeCount,bonds:gpu.bondCount,
    steps:gpu.stepCount,expectedFractureEvents:events,brokenBonds:broken,
    maxError,damageError,activeMismatches,invalid,
    synchronizedMs:completedMs,readbackMs,
    trackedGpuBytes:gpu.gpu.bytesAllocated(),
    passes:invalid===0&&activeMismatches===0&&maxError<=.006&&damageError<=.006})
  }finally{gpu.dispose()}
 }
 if(result)result.textContent=JSON.stringify({
  ok:results.every(x=>x.passes),adapter,
  numericalModel:'cohesive 3D Hookean axial bonds with strain damage and Handwired GPU integration',
  cpuOracle:'independently handwritten double-accumulating forces and fracture reference',
  cases:results,hardwareExpected:true,
  warning:'No finite-element continuum, measured material parameters, dynamic mesh reconstruction or scientific stress certification',
 })
}catch(e){if(result)result.textContent=JSON.stringify({ok:false,error:String(e)})}

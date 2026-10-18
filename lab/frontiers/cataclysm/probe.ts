import {CataclysmSphGpu} from './solver.ts'
import {densityPressureCpu,makeFluidBlock,stepSphCpu,sphDefaults,
 sphDiagnostics,type SphSettings,type SphState} from './model.ts'

const readout=document.getElementById('result')
type Scenario={name:string,initial:SphState,settings:SphSettings,steps:number}
const makeBlock=(nx:number,ny:number,nz:number,velocity=false):SphState=>{
 const initial=makeFluidBlock(nx,ny,nz)
 if(velocity){
  for(let i=0;i<initial.positions.length/4;i++){
   const p=i*4
   initial.velocities[p]=Math.fround((i%7-3)*.13)
   initial.velocities[p+1]=Math.fround((i%5-2)*.08)
   initial.velocities[p+2]=Math.fround((i%11-5)*.06)
  }
 }
 return initial
}
const scenarios:Scenario[]=[
 {name:'64-particle-3d-pressure',initial:makeBlock(4,4,4),
  settings:{...sphDefaults,restDensity:1.85,stiffness:12},steps:1},
 {name:'80-particle-viscous-flow',initial:makeBlock(5,4,4,true),
  settings:{...sphDefaults,restDensity:2,stiffness:9,viscosity:.12},steps:4},
 {name:'120-particle-gravity-evolution',initial:makeBlock(5,6,4,true),
  settings:{...sphDefaults,restDensity:2.1,stiffness:9},steps:8},
]
const results=[]
let adapter=''
try{
 for(const scenario of scenarios){
  const fieldReference=densityPressureCpu(scenario.initial,scenario.settings)
  let cpu=scenario.initial
  const baseline=sphDiagnostics(cpu,scenario.settings)
  for(let k=0;k<scenario.steps;k++)
   cpu=stepSphCpu(cpu,scenario.settings)
  const gpu=await CataclysmSphGpu.create(scenario.initial,scenario.settings)
  try{
   adapter=gpu.gpu.adapter.name
   const first=performance.now()
   gpu.step()
   await gpu.flush()
   const initialField=await gpu.readDensityPressure()
   for(let k=1;k<scenario.steps;k++)gpu.step()
   await gpu.flush()
   const synchronizedMs=performance.now()-first
   const readStart=performance.now(),actual=await gpu.read()
   const readbackMs=performance.now()-readStart
   let stateError=0,densityError=0,pressureError=0,nonfinite=0
   for(const axis of ['positions','velocities'] as const)
    for(let i=0;i<cpu[axis].length;i++){
     const observed=actual[axis][i]!
     if(!Number.isFinite(observed))nonfinite++
     stateError=Math.max(stateError,Math.abs(observed-cpu[axis][i]!))
    }
   for(let i=0;i<initialField.density.length;i++){
    const rho=initialField.density[i]!,p=initialField.pressure[i]!
    if(!Number.isFinite(rho)||!Number.isFinite(p))nonfinite++
    densityError=Math.max(densityError,Math.abs(rho-fieldReference.density[i]!))
    pressureError=Math.max(pressureError,Math.abs(p-fieldReference.pressure[i]!))
   }
   const computed=sphDiagnostics(actual,scenario.settings)
   const massError=Math.abs(computed.mass-baseline.mass)
   const passes=stateError<.012&&densityError<.015&&pressureError<.12&&
    nonfinite===0&&massError===0
   results.push({name:scenario.name,particles:gpu.count,
    steps:gpu.stepCount,passes,nonfinite,massError,
    maxStateError:stateError,maxDensityError:densityError,
    maxPressureError:pressureError,pressureMaximum:Math.max(...initialField.pressure),
    synchronizedMs,readbackMs,trackedGpuBytes:gpu.gpu.bytesAllocated(),
    expected:'independent 3D CPU O(N²) SPH with Gaussian-equivalent poly6/spiky/viscosity kernels'})
  }finally{gpu.dispose()}
 }
 if(readout)readout.textContent=JSON.stringify({ok:results.every(x=>x.passes),
  adapter,algorithm:'3D WCSPH on actual WebGPU linked-cell neighborhood grid',
  handwired:'compiled equation-of-state pressure Circuit in WGSL',
  cpuOracle:'independent all-pairs SPH density, pressure and viscosity reference',
  results,hardwareExpected:true,
  caveat:'WCSPH numerical model; no measured tsunami fidelity or free-surface reconstruction'})
}catch(e){if(readout)readout.textContent=JSON.stringify({ok:false,error:String(e)})}

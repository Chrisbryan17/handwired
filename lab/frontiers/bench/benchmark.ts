import {makeCollision,stepUniverseCpu,universeDiagnostics,STRIDE} from '../universe/model.ts'
import {UniverseGpuSolver} from '../universe/solver.ts'
import {makeGenesis,stepGenesisCpu,genesisDefaults} from '../genesis/model.ts'
import {GenesisGpuSolver} from '../genesis/solver.ts'

type NumericError={maxAbsoluteError:number,rootMeanSquareError:number,invalidCount:number}
function compare(expected:Float32Array,actual:Float32Array):NumericError{
  if(expected.length!==actual.length)throw new Error('Benchmark reference shape mismatch')
  let maxAbsoluteError=0,squared=0,invalidCount=0
  for(let i=0;i<expected.length;i++){
    const a=expected[i]!,b=actual[i]!
    if(!Number.isFinite(b)){invalidCount++;continue}
    const d=Math.abs(a-b)
    maxAbsoluteError=Math.max(maxAbsoluteError,d)
    squared+=d*d
  }
  return {maxAbsoluteError,rootMeanSquareError:Math.sqrt(squared/expected.length),
    invalidCount}
}
const now=()=>performance.now()
export type FrontierBenchmarkCase={
  solver:'newtonian-direct-nbody'|'grayscott-9point'
  workload:number,steps:number,adapter:string
  compilationAndInitializationMs:number,cpuReferenceMs:number,
  gpuSynchronizedComputationMs:number,gpuFinalReadbackMs:number,
  gpuMemoryTrackedBytes:number,calculatedCpuOverGpuRatio:number,
  numerical:NumericError,energyRelativeDrift?:number,momentumError?:number
}
export async function benchmarkUniverse(count:number,steps:number):Promise<FrontierBenchmarkCase>{
  const initial=makeCollision(count,37)
  const compilationStart=now()
  const gpu=await UniverseGpuSolver.create(initial,{gravity:.55,softening:.045})
  const compilationAndInitializationMs=now()-compilationStart
  try{
    gpu.step(3);await gpu.flush();gpu.reset(initial);await gpu.flush()
    const cpuStart=now()
    let reference=initial
    for(let n=0;n<steps;n++)reference=stepUniverseCpu(reference,{gravity:.55,softening:.045})
    const cpuReferenceMs=now()-cpuStart
    const gpuStart=now()
    // At most 64 passes per submission. Queue completion is synchronized, so
    // the measured interval is NOT merely host command submission latency.
    let remaining=steps
    while(remaining>0){const batch=Math.min(64,remaining);gpu.step(batch);remaining-=batch}
    await gpu.flush()
    const gpuSynchronizedComputationMs=now()-gpuStart
    const readStart=now()
    const calculated=await gpu.read()
    const gpuFinalReadbackMs=now()-readStart
    const numerical=compare(reference,calculated)
    let initialMomentum=universeDiagnostics(initial).momentum
    const final=universeDiagnostics(calculated)
    const initialEnergy=universeDiagnostics(initial).total
    const energyRelativeDrift=Math.abs((final.total-initialEnergy)/initialEnergy)
    let momentumError=0
    for(let i=0;i<3;i++)momentumError=Math.max(momentumError,
      Math.abs(final.momentum[i]!-initialMomentum[i]!))
    if(numerical.invalidCount>0)throw new Error('nonfinite GPU N-body state')
    return {solver:'newtonian-direct-nbody',workload:count,steps,adapter:gpu.gpu.adapter.name,
      compilationAndInitializationMs,cpuReferenceMs,gpuSynchronizedComputationMs,
      gpuFinalReadbackMs,gpuMemoryTrackedBytes:gpu.gpu.bytesAllocated(),
      calculatedCpuOverGpuRatio:cpuReferenceMs/gpuSynchronizedComputationMs,
      numerical,energyRelativeDrift,momentumError}
  }finally{gpu.dispose()}
}
export async function benchmarkGenesis(side:number,steps:number):Promise<FrontierBenchmarkCase>{
  const initial=makeGenesis(side,side,17)
  const compilationStart=now()
  const gpu=await GenesisGpuSolver.create(initial,genesisDefaults)
  const compilationAndInitializationMs=now()-compilationStart
  try{
    gpu.step(3);await gpu.gpu.device.queue.onSubmittedWorkDone()
    gpu.reset(initial);await gpu.gpu.device.queue.onSubmittedWorkDone()
    const cpuStart=now()
    let reference=initial
    for(let n=0;n<steps;n++)reference=stepGenesisCpu(reference,genesisDefaults)
    const cpuReferenceMs=now()-cpuStart
    const gpuStart=now()
    let remaining=steps
    while(remaining>0){const batch=Math.min(64,remaining);gpu.step(batch);remaining-=batch}
    await gpu.gpu.device.queue.onSubmittedWorkDone()
    const gpuSynchronizedComputationMs=now()-gpuStart
    const readStart=now()
    const calculated=await gpu.read()
    const gpuFinalReadbackMs=now()-readStart
    const numerical=compare(reference.state,calculated.state)
    if(numerical.invalidCount>0)throw new Error('nonfinite GPU chemical state')
    return {solver:'grayscott-9point',workload:side*side,steps,
      adapter:gpu.gpu.adapter.name,compilationAndInitializationMs,cpuReferenceMs,
      gpuSynchronizedComputationMs,gpuFinalReadbackMs,
      gpuMemoryTrackedBytes:gpu.gpu.bytesAllocated(),
      calculatedCpuOverGpuRatio:cpuReferenceMs/gpuSynchronizedComputationMs,numerical}
  }finally{gpu.dispose()}
}
export async function runBenchmarks(){
  const cases:FrontierBenchmarkCase[]=[]
  for(const n of [128,256,512])cases.push(await benchmarkUniverse(n,40))
  for(const side of [96,192,384]){
    const steps=side===384?12:40
    cases.push(await benchmarkGenesis(side,steps))
  }
  const adapters=new Set(cases.map(item=>item.adapter))
  if(adapters.size!==1)throw new Error('Benchmark had inconsistent GPU adapters')
  return {schemaVersion:1,recordedAt:new Date().toISOString(),adapter:[...adapters][0],
    metrics:{
      gpuTiming:'wall-clock queue submission plus synchronized device completion',
      cpuTiming:'independently handwritten deterministic CPU reference, matching step counts',
      excluded:'pipeline creation, cold shader compilation, initial upload, final readback, rendering',
      readback:'separate explicit final staging buffer copy and map',
      precision:'WGSL f32 versus JS arithmetic with f32 state storage',
      comparability:'different CPU and GPU arithmetic/order; measured ratio is not pure GPU hardware throughput',
      caveat:'single trial per workload; no error bars; thermals, scheduling and variance unquantified',
    },
    cases}
}
const status=document.getElementById('status')
const result=document.getElementById('result')
try{
  const report=await runBenchmarks()
  if(result)result.textContent=JSON.stringify({ok:true,...report})
  if(status)status.textContent='FINISHED'
}catch(error){
  if(result)result.textContent=JSON.stringify({ok:false,error:String(error instanceof Error?error.stack:error)})
  if(status)status.textContent='FAILED'
}

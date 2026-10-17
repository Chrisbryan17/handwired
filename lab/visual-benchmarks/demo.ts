import {ResidentFluidGpu} from '../visual-fluid/gpu-resident.ts'
import {createFluid,fluidAdvectionInputs,projectFluid,type Fluid} from '../visual-fluid/physics.ts'
import {handwrittenFluidStep,buildFluidCircuit,FLUID_SOURCE_SIZE} from '../visual-fluid/model.ts'
import {ResidentClothGpu} from '../visual-cloth/gpu-resident.ts'
import {makeCloth,clothSources,projectCloth,type Cloth,type ClothInteraction} from '../visual-cloth/physics.ts'
import {handwrittenClothStep,buildClothCircuit,CLOTH_SOURCE_SIZE} from '../visual-cloth/model.ts'

type Measurement={workload:'Fluid'|'Cloth',elements:number,steps:number,
  cpuMs:number,handwiredGpuMs:number,handwrittenGpuMs:number,
  gpuCompileMs:number,handwrittenCompileMs:number,
  parityMaxError:number,parityRms:number,ok:boolean,readbackIncluded:boolean}
const params=new URLSearchParams(location.search),smoke=params.get('smoke')==='1'
const status=document.querySelector<HTMLElement>('#status')!
const rows=document.querySelector<HTMLTableSectionElement>('#rows')!
const stepSelect=document.querySelector<HTMLSelectElement>('#steps')!
const runButton=document.querySelector<HTMLButtonElement>('#run')!
const put=(value:string)=>{status.textContent=value}
const fCircuit=buildFluidCircuit(),cCircuit=buildClothCircuit()
function median(values:number[]){const s=[...values].sort((a,b)=>a-b);return s[Math.floor(s.length/2)]!}
function referenceFluid(input:Fluid){
  const sources=fluidAdvectionInputs(input),width=fCircuit.circuit.sourceSize+fCircuit.circuit.operations.length
  const values=new Float32Array(input.width*input.height*width)
  for(let i=0;i<input.width*input.height;i++){
    const out=handwrittenFluidStep(sources.subarray(i*FLUID_SOURCE_SIZE,(i+1)*FLUID_SOURCE_SIZE))
    for(let c=0;c<5;c++)values[i*width+fCircuit.outputWires[c]!]=out[c]!
  }
  return projectFluid(input,values,width,fCircuit.outputWires)
}
function referenceCloth(input:Cloth){
  const interaction:ClothInteraction={dragged:null,x:.5,y:.5,wind:1.2}
  const sources=clothSources(input,interaction),width=cCircuit.circuit.sourceSize+cCircuit.circuit.operations.length
  const values=new Float32Array(input.columns*input.rows*width)
  for(let i=0;i<input.columns*input.rows;i++){
    const out=handwrittenClothStep(sources.subarray(i*CLOTH_SOURCE_SIZE,(i+1)*CLOTH_SOURCE_SIZE))
    values[i*width+cCircuit.outputWires[0]]=out[0]!
    values[i*width+cCircuit.outputWires[1]]=out[1]!
  }
  return projectCloth(input,values,width,cCircuit.outputWires,interaction)
}
function parity(left:Float32Array,right:Float32Array){
  if(left.length!==right.length)throw new Error('GPU implementations returned different output dimensions')
  let maxError=0,sum=0,nonfinite=0
  for(let i=0;i<left.length;i++){
    const a=left[i]!,b=right[i]!
    if(!Number.isFinite(a)||!Number.isFinite(b))nonfinite++
    const delta=Math.abs(a-b);maxError=Math.max(maxError,delta);sum+=delta*delta
  }
  return {maxError,rms:Math.sqrt(sum/left.length),nonfinite}
}
async function benchFluid(device:GPUDevice,width:number,height:number,steps:number,trials:number):Promise<Measurement>{
  const initial=createFluid(width,height),N=width*height
  const hwStart=performance.now(),hw=await ResidentFluidGpu.create(width,height,device,'handwired')
  const hwCompile=performance.now()-hwStart
  const baselineStart=performance.now(),hand=await ResidentFluidGpu.create(width,height,device,'handwritten')
  const baselineCompile=performance.now()-baselineStart
  try{
    const cpuTimes:number[]=[],hwTimes:number[]=[],handTimes:number[]=[]
    for(let run=0;run<trials;run++){
      let cpu=createFluid(width,height)
      const startedCpu=performance.now()
      for(let i=0;i<steps;i++)cpu=referenceFluid(cpu)
      cpuTimes.push(performance.now()-startedCpu)
      for(const [gpu,times] of [[hw,hwTimes],[hand,handTimes]] as const){
        gpu.load(initial)
        const started=performance.now()
        for(let i=0;i<steps;i++)await gpu.step()
        await gpu.read()
        times.push(performance.now()-started)
      }
    }
    hw.load(initial);hand.load(initial)
    for(let i=0;i<steps;i++){await hw.step();await hand.step()}
    const match=parity(await hw.read(),await hand.read())
    return {workload:'Fluid',elements:N,steps,
      cpuMs:median(cpuTimes)/steps,handwiredGpuMs:median(hwTimes)/steps,
      handwrittenGpuMs:median(handTimes)/steps,gpuCompileMs:hwCompile,
      handwrittenCompileMs:baselineCompile,parityMaxError:match.maxError,parityRms:match.rms,
      readbackIncluded:true,ok:match.nonfinite===0&&match.maxError<.002}
  }finally{hw.destroy();hand.destroy()}
}
async function benchCloth(device:GPUDevice,columns:number,rows:number,steps:number,trials:number):Promise<Measurement>{
  const initial=makeCloth(columns,rows),N=columns*rows
  const hwStart=performance.now(),hw=await ResidentClothGpu.create(initial,device,'handwired')
  const hwCompile=performance.now()-hwStart
  const baselineStart=performance.now(),hand=await ResidentClothGpu.create(initial,device,'handwritten')
  const baselineCompile=performance.now()-baselineStart
  try{
    const cpuTimes:number[]=[],hwTimes:number[]=[],handTimes:number[]=[]
    for(let run=0;run<trials;run++){
      let cpu=makeCloth(columns,rows)
      const startedCpu=performance.now()
      for(let i=0;i<steps;i++)cpu=referenceCloth(cpu)
      cpuTimes.push(performance.now()-startedCpu)
      for(const [gpu,times] of [[hw,hwTimes],[hand,handTimes]] as const){
        gpu.load(initial)
        const started=performance.now()
        for(let i=0;i<steps;i++)await gpu.step({wind:1.2})
        await gpu.read()
        times.push(performance.now()-started)
      }
    }
    hw.load(initial);hand.load(initial)
    for(let i=0;i<steps;i++){await hw.step({wind:1.2});await hand.step({wind:1.2})}
    const match=parity(await hw.read(),await hand.read())
    return {workload:'Cloth',elements:N,steps,
      cpuMs:median(cpuTimes)/steps,handwiredGpuMs:median(hwTimes)/steps,
      handwrittenGpuMs:median(handTimes)/steps,gpuCompileMs:hwCompile,
      handwrittenCompileMs:baselineCompile,parityMaxError:match.maxError,parityRms:match.rms,
      readbackIncluded:true,ok:match.nonfinite===0&&match.maxError<.002}
  }finally{hw.destroy();hand.destroy()}
}
function render(result:Measurement){
  const row=document.createElement('tr')
  const items=[result.workload,result.elements.toLocaleString(),
    result.cpuMs.toFixed(3),result.handwiredGpuMs.toFixed(3),
    result.handwrittenGpuMs.toFixed(3),result.gpuCompileMs.toFixed(1),
    result.ok?'PASS · '+result.parityMaxError.toExponential(2):
      'FAIL · '+result.parityMaxError.toExponential(2)]
  for(const [index,value] of items.entries()){
    const cell=document.createElement('td');cell.textContent=value
    if(index===6)cell.className=result.ok?'pass':'fail'
    row.appendChild(cell)
  }
  rows.appendChild(row)
}
async function run(){
  runButton.disabled=true
  rows.replaceChildren()
  const steps=smoke?8:Number(stepSelect.value),trials=smoke?2:3
  const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'})
  if(!adapter)throw new Error('WebGPU adapter unavailable; cannot make hardware benchmark claims')
  const device=await adapter.requestDevice()
  document.querySelector<HTMLElement>('#device')!.textContent=
    'GPU '+[adapter.info.vendor,adapter.info.architecture,adapter.info.device].join(' / ')
  const cases:readonly (readonly ['fluid'|'cloth',number,number])[]=smoke?
    [['fluid',24,16],['cloth',15,11]]:
    [['fluid',24,16],['fluid',48,32],['fluid',64,40],
      ['cloth',15,11],['cloth',25,17],['cloth',33,23]]
  const results:Measurement[]=[]
  try{
    for(const [name,width,height] of cases){
      put('Running '+name+' '+width+' × '+height+' ('+steps+' steps × '+trials+' trials)…')
      const measurement=name==='fluid'?
        await benchFluid(device,width,height,steps,trials):
        await benchCloth(device,width,height,steps,trials)
      results.push(measurement)
      render(measurement)
    }
    const ok=results.every(x=>x.ok)
    put(ok?'GPU/CPU benchmark comparisons completed successfully.':'A GPU numeric comparison failed; see results.')
    const report={ok,mode:'gpu',frames:8,count:results.length,checked:results.length,
      mismatches:results.filter(x=>!x.ok).length,maxError:Math.max(...results.map(x=>x.parityMaxError)),
      browserAdapter:adapter.info,steps,trials,results}
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
    document.body.dataset.status=ok?'pass':'fail'
  }finally{device.destroy();runButton.disabled=false}
}
runButton.addEventListener('click',()=>{void run().catch(fail)})
function fail(error:unknown){
  const message=String(error);put('Benchmark failed: '+message)
  document.body.dataset.status='fail'
  document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:message})
  runButton.disabled=false
}
if(smoke)void run().catch(fail)

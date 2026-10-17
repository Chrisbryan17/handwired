import {createWebGpuEvaluator,type WebGpuCircuitEvaluator} from '../../src/runtime/webgpu.ts'
import {buildFluidCircuit,FLUID_SOURCE_SIZE,FLUID_CHANNELS} from './model.ts'
import {createFluid,fluidAdvectionInputs,projectFluid,cpuFluidFrame,injectFluid,type Fluid} from './physics.ts'
import {ResidentFluidGpu,type FluidPointer} from './gpu-resident.ts'
const params=new URLSearchParams(location.search),smoke=params.get('smoke')==='1'
const {circuit,outputWires}=buildFluidCircuit(),width=circuit.sourceSize+circuit.operations.length
const size=document.querySelector<HTMLSelectElement>('#size')!
const strategy=document.querySelector<HTMLSelectElement>('#strategy')!
const engine=document.querySelector<HTMLSelectElement>('#engine')!
const wantsResident=params.get('resident')==='1'
engine.value=smoke&&!wantsResident?'hybrid':'resident'
const brush=document.querySelector<HTMLInputElement>('#brush')!
const cpuCanvas=document.querySelector<HTMLCanvasElement>('#cpu')!,gpuCanvas=document.querySelector<HTMLCanvasElement>('#gpu')!
const cpuCtx=cpuCanvas.getContext('2d')!,gpuCtx=gpuCanvas.getContext('2d')!
const status=document.querySelector<HTMLElement>('#status')!
const put=(id:string,val:string)=>{document.getElementById(id)!.textContent=val}
let cpu=createFluid(),gpu=createFluid(),backend:WebGpuCircuitEvaluator|null=null
let resident:ResidentFluidGpu|null=null,pendingImpulse:FluidPointer|undefined
let resetRequested=true,paused=false,frames=0,checked=0,mismatches=0,maxError=0,cpuMs=0,gpuMs=0
const buffer=document.createElement('canvas'),bufferContext=buffer.getContext('2d')!
const clamp=(x:number)=>Math.max(0,Math.min(1,x))
function draw(canvas:HTMLCanvasElement,ctx:CanvasRenderingContext2D,fluid:Fluid) {
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(2,devicePixelRatio||1)
  const px=Math.max(1,Math.round(rect.width*ratio)),py=Math.max(1,Math.round(rect.height*ratio))
  if(canvas.width!==px||canvas.height!==py){canvas.width=px;canvas.height=py}
  ctx.setTransform(ratio,0,0,ratio,0,0)
  if(buffer.width!==fluid.width||buffer.height!==fluid.height){
    buffer.width=fluid.width;buffer.height=fluid.height
  }
  const pixels=bufferContext.createImageData(fluid.width,fluid.height)
  for(let i=0;i<fluid.width*fluid.height;i++){
    const at=i*FLUID_CHANNELS,d=i*4
    const r=1-Math.exp(-Math.max(0,fluid.fields[at+2]!)*1.15)
    const g=1-Math.exp(-Math.max(0,fluid.fields[at+3]!)*1.15)
    const b=1-Math.exp(-Math.max(0,fluid.fields[at+4]!)*1.15)
    const brightness=Math.min(1,(r+g+b)*.22)
    pixels.data[d]=Math.round(6+235*r+40*brightness)
    pixels.data[d+1]=Math.round(6+210*g+32*brightness)
    pixels.data[d+2]=Math.round(22+220*b+40*brightness)
    pixels.data[d+3]=255
  }
  bufferContext.putImageData(pixels,0,0)
  ctx.fillStyle='#060817';ctx.fillRect(0,0,rect.width,rect.height)
  ctx.imageSmoothingEnabled=true
  ctx.drawImage(buffer,0,0,rect.width,rect.height)
}
async function reset(){
  backend?.destroy();backend=null;resident?.destroy();resident=null;pendingImpulse=undefined
  const selected=Number(size.value),height=selected===32?24:selected===64?40:32
  cpu=createFluid(selected,height);gpu=createFluid(selected,height)
  frames=0;checked=0;mismatches=0;maxError=0
  status.textContent='Compiling neural advection and fluid stencil pipelines…'
  if(engine.value==='resident') {
    try{
      resident=await ResidentFluidGpu.create(cpu.width,cpu.height)
      resident.load(gpu)
      status.textContent='Entire neural fluid solver GPU-resident. Drag to stir the dye.'
    }catch(error){
      if(smoke&&wantsResident)throw error
      status.textContent='Resident GPU unavailable, using hybrid CPU fallback: '+String(error)
      engine.value='hybrid'
    }
  }
  if(resident===null) {
    backend=await createWebGpuEvaluator(circuit,{
      mode:smoke?'gpu':'auto',capacity:cpu.width*cpu.height,
      strategy:strategy.value as 'specialized'|'monolithic'|'layers'})
    status.textContent=backend.mode==='gpu'
      ? 'Hybrid: neural bilinear advection on WebGPU, pressure solver on CPU.'
      : 'CPU fallback: '+(backend.fallbackReason??'WebGPU unavailable')
  }
  put('checked-label',resident?'FIELD VALUES CHECKED':'CIRCUIT WIRES CHECKED')
  put('errors-label',resident?'POSE/FIELD DIFFS (>0.025)':'GPU MISMATCHES (>2e−5)')
  put('cells',String(cpu.width*cpu.height));put('ops',String(circuit.operations.length))
  resetRequested=false
}
function verify(inputs:Float32Array,values:Float32Array){
  for(let i=0;i<gpu.width*gpu.height;i++){
    const expected=circuit.evaluateReference(inputs.subarray(i*FLUID_SOURCE_SIZE,(i+1)*FLUID_SOURCE_SIZE))
    for(let k=0;k<width;k++){
      const delta=Math.abs(expected[k]!-values[i*width+k]!)
      if(!Number.isFinite(delta)||delta>2e-5)mismatches++
      maxError=Math.max(maxError,delta);checked++
    }
  }
}
async function tick(){
  if(resetRequested||(backend===null&&resident===null))await reset()
  if(paused)return
  const startedCpu=performance.now(),reference=cpuFluidFrame(cpu)
  cpu=reference.next;cpuMs=performance.now()-startedCpu
  if(resident!==null){
    const startedGpu=performance.now()
    await resident.step(pendingImpulse)
    pendingImpulse=undefined
    const values=await resident.read()
    gpuMs=performance.now()-startedGpu
    gpu={...gpu,fields:values,step:gpu.step+1}
    if(smoke||frames%5===0){
      for(let i=0;i<values.length;i++){
        const delta=Math.abs(values[i]!-cpu.fields[i]!)
        if(!Number.isFinite(delta)||delta>.025)mismatches++
        maxError=Math.max(maxError,delta);checked++
      }
    }
  }else{
    const input=fluidAdvectionInputs(gpu),startedGpu=performance.now()
    const values=await backend!.evaluate(input)
    gpuMs=performance.now()-startedGpu
    if(smoke||frames%5===0)verify(input,values)
    gpu=projectFluid(gpu,values,width,outputWires)
  }
  frames++
  draw(cpuCanvas,cpuCtx,cpu);draw(gpuCanvas,gpuCtx,gpu)
  put('checked',checked.toLocaleString());put('errors',String(mismatches))
  put('cpu-info','CPU '+cpuMs.toFixed(1)+' ms/frame · step '+frames)
  put('gpu-info',(resident?'FULL GPU':backend!.mode.toUpperCase())+' '+gpuMs.toFixed(1)+' ms/frame · max error '+maxError.toExponential(2))
  document.body.dataset.status=mismatches===0?'pass':'fail'
  if(smoke&&frames>=8){
    const actualMode=resident?'gpu':backend!.mode
    const result={ok:mismatches===0&&actualMode==='gpu',mode:actualMode,resident:resident!==null,
      frames,count:gpu.width*gpu.height,operations:circuit.operations.length,checked,
      mismatches,maxError,cpuMs,gpuMs}
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(result)
    document.body.dataset.status=result.ok?'pass':'fail'
    paused=true
  }
}
async function loop(){
  try{await tick()}catch(error){
    status.textContent='Fluid experiment failed: '+String(error)
    document.body.dataset.status='fail'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
    paused=true
  }
  if(!(smoke&&paused))requestAnimationFrame(()=>{void loop()})
}
for(const canvas of [cpuCanvas,gpuCanvas]){
  let previous:{x:number,y:number}|null=null
  canvas.addEventListener('pointerdown',event=>{canvas.setPointerCapture(event.pointerId);previous=null})
  canvas.addEventListener('pointermove',event=>{
    if(event.buttons!==1)return
    const rect=canvas.getBoundingClientRect()
    const x=clamp((event.clientX-rect.left)/rect.width),y=clamp((event.clientY-rect.top)/rect.height)
    const dx=previous?x-previous.x:0,dy=previous?y-previous.y:0
    const force=Number(brush.value)
    injectFluid(cpu,x,y,dx,dy,force)
    if(resident!==null)pendingImpulse={x,y,dx,dy,strength:force}
    else injectFluid(gpu,x,y,dx,dy,force)
    previous={x,y}
  })
  canvas.addEventListener('pointerup',()=>{previous=null})
  canvas.addEventListener('pointercancel',()=>{previous=null})
}
document.querySelector<HTMLButtonElement>('#pause')!.onclick=()=>{
  paused=!paused;put('pause',paused?'RESUME':'PAUSE')}
document.querySelector<HTMLButtonElement>('#reset')!.onclick=()=>{
  resetRequested=true;paused=false;put('pause','PAUSE')}
size.onchange=()=>{resetRequested=true}
strategy.onchange=()=>{resetRequested=true}
engine.onchange=()=>{resetRequested=true}
void loop()

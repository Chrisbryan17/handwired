import {createWebGpuEvaluator,type WebGpuCircuitEvaluator} from '../../src/runtime/webgpu.ts'
import {buildClothCircuit,CLOTH_SOURCE_SIZE} from './model.ts'
import {makeCloth,clothSources,projectCloth,cpuClothFrame,tearNear,nearestNode,
  type Cloth,type ClothInteraction} from './physics.ts'
import {ResidentClothGpu} from './gpu-resident.ts'
const query=new URLSearchParams(location.search),smoke=query.get('smoke')==='1'
const {circuit,outputWires}=buildClothCircuit(),width=circuit.sourceSize+circuit.operations.length
const tool=document.querySelector<HTMLSelectElement>('#tool')!
const wind=document.querySelector<HTMLInputElement>('#wind')!
const strategy=document.querySelector<HTMLSelectElement>('#strategy')!
const engine=document.querySelector<HTMLSelectElement>('#engine')!
const wantsResident=query.get('resident')==='1'
engine.value=smoke&&!wantsResident?'hybrid':'resident'
const canvasCpu=document.querySelector<HTMLCanvasElement>('#cpu')!
const canvasGpu=document.querySelector<HTMLCanvasElement>('#gpu')!
const ctxCpu=canvasCpu.getContext('2d')!,ctxGpu=canvasGpu.getContext('2d')!
const status=document.querySelector<HTMLElement>('#status')!
const put=(id:string,value:string)=>{document.getElementById(id)!.textContent=value}
let cpu=makeCloth(),gpu=makeCloth(),backend:WebGpuCircuitEvaluator|null=null
let resident:ResidentClothGpu|null=null,pendingCut=false
let needsReset=true,paused=false,frames=0,checks=0,mismatches=0,maxError=0,cpuMs=0,gpuMs=0
const interact:ClothInteraction={dragged:null,x:.5,y:.5,wind:1.2}
function canvasSize(canvas:HTMLCanvasElement,context:CanvasRenderingContext2D){
  const rect=canvas.getBoundingClientRect(),scale=Math.min(2,devicePixelRatio||1)
  if(canvas.width!==Math.round(rect.width*scale)||canvas.height!==Math.round(rect.height*scale)){
    canvas.width=Math.max(1,Math.round(rect.width*scale))
    canvas.height=Math.max(1,Math.round(rect.height*scale))
  }
  context.setTransform(scale,0,0,scale,0,0)
  return {width:rect.width,height:rect.height}
}
function draw(canvas:HTMLCanvasElement,ctx:CanvasRenderingContext2D,cloth:Cloth) {
  const bounds=canvasSize(canvas,ctx),W=bounds.width,H=bounds.height
  ctx.fillStyle='#071221';ctx.fillRect(0,0,W,H)
  const sphereX=.5*W,sphereY=.78*H,sphereR=.14*Math.min(W,H)
  const glow=ctx.createRadialGradient(sphereX,sphereY,sphereR*.3,sphereX,sphereY,sphereR*1.5)
  glow.addColorStop(0,'#b6deeb44');glow.addColorStop(.7,'#538ab055');glow.addColorStop(1,'#538ab000')
  ctx.fillStyle=glow;ctx.beginPath();ctx.arc(sphereX,sphereY,sphereR*1.5,0,Math.PI*2);ctx.fill()
  ctx.strokeStyle='#78a9c966';ctx.lineWidth=2
  ctx.beginPath();ctx.arc(sphereX,sphereY,sphereR,0,Math.PI*2);ctx.stroke()
  const s=cloth.state,columns=cloth.columns
  // Draw shaded quads, then structural threads to make deformation legible.
  for(let y=0;y<cloth.rows-1;y++)for(let x=0;x<columns-1;x++){
    const i=y*columns+x,a=i*4,b=(i+1)*4,c=(i+1+columns)*4,d=(i+columns)*4
    ctx.beginPath();ctx.moveTo(s[a]!*W,s[a+1]!*H);ctx.lineTo(s[b]!*W,s[b+1]!*H)
    ctx.lineTo(s[c]!*W,s[c+1]!*H);ctx.lineTo(s[d]!*W,s[d+1]!*H);ctx.closePath()
    const alpha=.12+(y/cloth.rows)*.1
    ctx.fillStyle='rgba(72,190,206,'+alpha.toFixed(3)+')'
    ctx.fill()
  }
  for(const edge of cloth.springs) {
    if(!edge.alive || Math.abs(edge.b-edge.a)!==1 && Math.abs(edge.b-edge.a)!==columns)continue
    const a=edge.a*4,b=edge.b*4
    const stretch=Math.hypot(s[b]!-s[a]!,s[b+1]!-s[a+1]!)/edge.rest
    ctx.strokeStyle=stretch>1.28?'#ffb878a0':'#88e3eb8f'
    ctx.lineWidth=stretch>1.28?1.25:.8
    ctx.beginPath();ctx.moveTo(s[a]!*W,s[a+1]!*H);ctx.lineTo(s[b]!*W,s[b+1]!*H);ctx.stroke()
  }
  ctx.fillStyle='#fff0b0'
  for(let i=0;i<cloth.pinned.length;i++)if(cloth.pinned[i]===1){
    ctx.beginPath();ctx.arc(s[i*4]!*W,s[i*4+1]!*H,2.6,0,Math.PI*2);ctx.fill()
  }
}
async function reset(){
  backend?.destroy();backend=null;resident?.destroy();resident=null;pendingCut=false
  cpu=makeCloth();gpu=makeCloth();frames=0;checks=0;mismatches=0;maxError=0
  interact.dragged=null;interact.wind=Number(wind.value)
  status.textContent='Compiling neural Verlet, springs and GPU constraint kernels…'
  if(engine.value==='resident'){
    try{
      resident=await ResidentClothGpu.create(gpu)
      resident.load(gpu)
      status.textContent='Full GPU cloth: spring forces, neural Verlet, collisions and tear masks.'
    }catch(error){
      if(smoke&&wantsResident)throw error
      engine.value='hybrid'
      status.textContent='Full GPU unavailable, falling back to hybrid: '+String(error)
    }
  }
  if(resident===null){
    backend=await createWebGpuEvaluator(circuit,{mode:smoke?'gpu':'auto',
      capacity:cpu.columns*cpu.rows,strategy:strategy.value as 'specialized'|'monolithic'|'layers'})
    status.textContent=backend.mode==='gpu'?'Hybrid WebGPU Verlet. Host computes forces and constraints.':
      'CPU fallback: '+(backend.fallbackReason??'No WebGPU device')
  }
  put('checked-label',resident?'VERTEX POSITIONS CHECKED':'CIRCUIT WIRES VERIFIED')
  put('mismatches-label',resident?'CPU POSE DIFFERENCES (>0.16)':'GPU MISMATCHES (>2e−5)')
  needsReset=false
  put('vertices',String(cpu.columns*cpu.rows));put('springs',String(cpu.springs.length))
}
function compare(input:Float32Array,values:Float32Array){
  const n=cpu.columns*cpu.rows
  for(let i=0;i<n;i++){
    const expected=circuit.evaluateReference(input.subarray(i*CLOTH_SOURCE_SIZE,(i+1)*CLOTH_SOURCE_SIZE))
    for(let j=0;j<width;j++){
      const delta=Math.abs(expected[j]!-values[i*width+j]!)
      if(!Number.isFinite(delta)||delta>2e-5)mismatches++
      maxError=Math.max(maxError,delta);checks++
    }
  }
}
async function tick(){
  if(needsReset||(backend===null&&resident===null))await reset()
  if(paused)return
  interact.wind=Number(wind.value)
  const startCpu=performance.now(),c=cpuClothFrame(cpu,interact)
  cpu=c.next;cpuMs=performance.now()-startCpu
  if(resident!==null){
    const startGpu=performance.now()
    await resident.step({dragged:interact.dragged,x:interact.x,y:interact.y,
      wind:interact.wind,cut:pendingCut,tearRadius:.025})
    pendingCut=false
    const values=await resident.read()
    gpuMs=performance.now()-startGpu
    gpu={...gpu,state:values,step:gpu.step+1}
    if(smoke||frames%5===0){
      for(let i=0;i<gpu.columns*gpu.rows;i++)for(const axis of [0,1]){
        const at=i*4+axis,delta=Math.abs(values[at]!-cpu.state[at]!)
        if(!Number.isFinite(delta)||delta>.16)mismatches++
        maxError=Math.max(maxError,delta);checks++
      }
    }
  }else{
    const inputs=clothSources(gpu,interact),startGpu=performance.now()
    const values=await backend!.evaluate(inputs)
    gpuMs=performance.now()-startGpu
    if(smoke||frames%5===0)compare(inputs,values)
    gpu=projectCloth(gpu,values,width,outputWires,interact)
  }
  frames++
  draw(canvasCpu,ctxCpu,cpu);draw(canvasGpu,ctxGpu,gpu)
  put('checked',checks.toLocaleString());put('mismatches',mismatches.toLocaleString())
  put('broken',String(gpu.springs.filter(e=>!e.alive).length))
  put('cpu-info','CPU '+cpuMs.toFixed(2)+'ms/frame · step '+frames)
  put('gpu-info',(resident?'FULL GPU':backend!.mode.toUpperCase())+' '+gpuMs.toFixed(2)+'ms/frame · max error '+maxError.toExponential(1))
  document.body.dataset.status=mismatches===0?'pass':'fail'
  if(smoke&&frames>=8){
    const actualMode=resident?'gpu':backend!.mode
    const result={ok:mismatches===0&&actualMode==='gpu',mode:actualMode,resident:resident!==null,frames,
      count:gpu.columns*gpu.rows,operations:circuit.operations.length,checked:checks,
      mismatches,maxError,cpuMs,gpuMs}
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(result)
    document.body.dataset.status=result.ok?'pass':'fail'
    paused=true
  }
}
async function loop(){
  try{await tick()}catch(error){
    status.textContent='Experiment failed: '+String(error)
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
    document.body.dataset.status='fail';paused=true
  }
  if(!(smoke&&paused))requestAnimationFrame(()=>{void loop()})
}
for(const canvas of [canvasCpu,canvasGpu]){
  const move=(event:PointerEvent)=>{
    const rect=canvas.getBoundingClientRect()
    interact.x=Math.max(.01,Math.min(.99,(event.clientX-rect.left)/rect.width))
    interact.y=Math.max(.01,Math.min(.99,(event.clientY-rect.top)/rect.height))
    if(event.buttons===1&&tool.value==='cut'){
      tearNear(cpu,interact.x,interact.y,.015)
      tearNear(gpu,interact.x,interact.y,.015)
      pendingCut=true
    }
  }
  canvas.addEventListener('pointerdown',event=>{
    canvas.setPointerCapture(event.pointerId);move(event)
    if(tool.value==='grab'){
      const chosen=nearestNode(canvas===canvasCpu?cpu:gpu,interact.x,interact.y)
      interact.dragged=chosen
    }else{
      tearNear(cpu,interact.x,interact.y,.024)
      tearNear(gpu,interact.x,interact.y,.024)
      pendingCut=true
    }
  })
  canvas.addEventListener('pointermove',move)
  canvas.addEventListener('pointerup',()=>{interact.dragged=null})
  canvas.addEventListener('pointercancel',()=>{interact.dragged=null})
}
document.querySelector<HTMLButtonElement>('#pause')!.onclick=()=>{
  paused=!paused;put('pause',paused?'RESUME':'PAUSE')}
document.querySelector<HTMLButtonElement>('#restart')!.onclick=()=>{
  needsReset=true;paused=false;put('pause','PAUSE')}
strategy.onchange=()=>{needsReset=true}
engine.onchange=()=>{needsReset=true}
void loop()

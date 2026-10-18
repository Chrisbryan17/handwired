import {LightGpuRenderer,lightDefaults} from './solver.ts'
const get=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T
const canvas=get<HTMLCanvasElement>('light-canvas')
const resolution=get<HTMLSelectElement>('resolution')
const depth=get<HTMLInputElement>('bounces')
const intensity=get<HTMLInputElement>('light')
const glass=get<HTMLInputElement>('ior')
const exposure=get<HTMLInputElement>('exposure')
const pause=get<HTMLButtonElement>('pause')
const error=get<HTMLElement>('error')
const output=(name:string,value:string)=>{get<HTMLElement>(name).textContent=value}
let engine:LightGpuRenderer|null=null,request=0,raf=0,halted=false
let frames=0,lastFrame=performance.now()
function displayControls(){
 output('resolution-label',resolution.value+' × '+Math.round(Number(resolution.value)*.75))
 output('bounces-label',depth.value)
 output('light-label',Number(intensity.value).toFixed(2))
 output('ior-label',Number(glass.value).toFixed(2))
 output('exposure-label',Number(exposure.value).toFixed(2))
 output('ray-depth',depth.value)
}
function fail(e:unknown){
 error.hidden=false
 error.textContent='WebGPU tracing interrupted: '+String(e instanceof Error?e.message:e)
 output('gpu-status','GPU UNAVAILABLE')
}
const settings=()=>({maxBounces:Number(depth.value),lightStrength:Number(intensity.value),
 ior:Number(glass.value),exposure:Number(exposure.value)})
function tick(now:number){
 if(!engine)return
 try{
  if(!halted && engine.samplesPerPixel<2000)engine.frame()
  frames++
  output('spp',engine.samplesPerPixel.toLocaleString())
  if(now-lastFrame>1000){
   output('fps',String(Math.round(frames*1000/(now-lastFrame))))
   frames=0;lastFrame=now
  }
  raf=requestAnimationFrame(tick)
 }catch(e){fail(e)}
}
async function start(){
 const generation=++request
 cancelAnimationFrame(raf);engine?.dispose();engine=null
 halted=false;pause.textContent='Pause';error.hidden=true
 output('gpu-status','COMPILING WEBGPU')
 const width=Number(resolution.value),height=Math.floor(width*.75)
 displayControls()
 try{
  const tracer=await LightGpuRenderer.create(canvas,width,height,{...lightDefaults,...settings()})
  if(request!==generation){tracer.dispose();return}
  engine=tracer
  output('pixels',(width*height).toLocaleString())
  output('bytes',(tracer.gpuBytes/1024).toFixed(1))
  output('adapter',tracer.adapterName)
  output('gpu-status','GPU PATH TRACING ACTIVE')
  output('verification','The GPU renders stochastic light paths. Separate reference tests and hardware verification are in the repository; browser display alone does not certify numerical agreement.')
  frames=0;lastFrame=performance.now()
  raf=requestAnimationFrame(tick)
 }catch(e){if(generation===request)fail(e)}
}
const update=()=>{
 displayControls()
 if(engine)try{engine.update(settings())}catch(e){fail(e)}
}
resolution.addEventListener('change',()=>{void start()})
for(const input of [depth,intensity,glass,exposure])input.addEventListener('input',update)
pause.addEventListener('click',()=>{halted=!halted;pause.textContent=halted?'Resume':'Pause'})
get<HTMLButtonElement>('restart').addEventListener('click',()=>{engine?.reset()})
get<HTMLButtonElement>('save').addEventListener('click',()=>{
 if(!engine)return
 const anchor=document.createElement('a')
 anchor.href=canvas.toDataURL('image/png')
 anchor.download='handwired-neural-light-'+engine.samplesPerPixel+'spp.png'
 anchor.click()
})
let touching=false,lastX=0,lastY=0
canvas.addEventListener('pointerdown',event=>{
 if(!engine)return
 touching=true;lastX=event.clientX;lastY=event.clientY
 canvas.setPointerCapture(event.pointerId)
})
canvas.addEventListener('pointermove',event=>{
 if(!touching||!engine)return
 const deltaX=event.clientX-lastX,deltaY=event.clientY-lastY
 lastX=event.clientX;lastY=event.clientY
 try{engine.update({yaw:Math.max(-2,Math.min(2,engine.settings.yaw+deltaX*.004)),
 pitch:Math.max(-1.1,Math.min(1.1,engine.settings.pitch+deltaY*.004))})}catch(e){fail(e)}
})
canvas.addEventListener('pointerup',()=>{touching=false})
canvas.addEventListener('pointercancel',()=>{touching=false})
window.addEventListener('pagehide',()=>{
 cancelAnimationFrame(raf);engine?.dispose()
})
void start()

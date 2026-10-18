import {makeCollision,stepUniverseCpu,universeDiagnostics,
  FIXED_DT,DEFAULT_G,DEFAULT_SOFTENING,STRIDE,
  type UniverseState} from './model.ts'
import {UniverseGpuSolver} from './solver.ts'
import {UniverseRenderer} from './renderer.ts'

const byId=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T
const canvas=byId<HTMLCanvasElement>('universe')
const population=byId<HTMLSelectElement>('population')
const gravity=byId<HTMLInputElement>('gravity')
const softening=byId<HTMLInputElement>('softening')
const zoom=byId<HTMLInputElement>('zoom')
const errorBox=byId<HTMLElement>('error')
const pauseButton=byId<HTMLButtonElement>('pause')
const verifyButton=byId<HTMLButtonElement>('verify')
let solver:UniverseGpuSolver|null=null
let viewer:UniverseRenderer|null=null
let initial:UniverseState=makeCollision(256)
let epoch=0,raf=0,paused=false,verifying=false,lastFPS=performance.now(),frames=0
const display=(id:string,value:string)=>{byId<HTMLElement>(id).textContent=value}
const failure=(err:unknown)=>{
  errorBox.hidden=false
  errorBox.textContent='GPU simulation stopped: '+String(err instanceof Error?err.message:err)
  display('device-chip','GPU ERROR')
}
function setParameters(){
  display('gravity-output',Number(gravity.value).toFixed(2))
  display('softening-output',Number(softening.value).toFixed(3))
  display('zoom-output',Number(zoom.value).toFixed(2))
  if(solver)solver.setParameters({gravity:Number(gravity.value),softening:Number(softening.value)})
  if(viewer)viewer.zoom=Number(zoom.value)
}
function frame(now:number) {
  if(!solver||!viewer)return
  try{
    viewer.render(!paused&&!verifying)
    frames++
    display('steps',solver.frame.toLocaleString())
    display('time',(solver.frame*FIXED_DT).toFixed(2))
    if(now-lastFPS>1000){
      display('fps',String(Math.round(frames*1000/(now-lastFPS))))
      lastFPS=now;frames=0
    }
    raf=requestAnimationFrame(frame)
  }catch(err){failure(err)}
}
async function begin(count:number){
  const run=++epoch
  cancelAnimationFrame(raf)
  viewer?.dispose();solver?.dispose()
  viewer=null;solver=null
  errorBox.hidden=true;paused=false;verifying=false;pauseButton.textContent='Pause'
  display('device-chip','COMPILING WEBGPU')
  display('verification-result','Run a synchronized differential check to validate this adapter.')
  initial=makeCollision(count)
  try{
    const engine=await UniverseGpuSolver.create(initial,{
      gravity:Number(gravity.value),softening:Number(softening.value),
    })
    if(run!==epoch){engine.dispose();return}
    const camera=await UniverseRenderer.create(engine,canvas)
    if(run!==epoch){camera.dispose();engine.dispose();return}
    solver=engine;viewer=camera;setParameters()
    display('population-output',count.toLocaleString())
    display('bodies',count.toLocaleString())
    display('memory',(engine.gpu.bytesAllocated()/1024).toFixed(1))
    display('adapter',engine.gpu.adapter.name)
    display('device-chip','WEBGPU COMPUTE ACTIVE')
    display('gpu-label','DIRECT O(N²) · FP32 · GPU-RESIDENT')
    lastFPS=performance.now();frames=0
    raf=requestAnimationFrame(frame)
  }catch(err){if(run===epoch)failure(err)}
}
async function verify(){
  if(!solver||!viewer||verifying)return
  verifying=true;paused=true;pauseButton.textContent='Resume'
  verifyButton.disabled=true
  display('verification-result','Synchronizing GPU and comparing independently calculated forces…')
  try{
    solver.reset(initial)
    solver.step()
    const gpuResult=await solver.read()
    const cpuResult=stepUniverseCpu(initial,solver.params)
    let maxError=0,nonfinite=0
    for(let i=0;i<initial.length/STRIDE;i++){
      for(const k of [0,1,2,4,5,6]){
        const index=i*STRIDE+k
        const value=gpuResult[index]!
        if(!Number.isFinite(value))nonfinite++
        maxError=Math.max(maxError,Math.abs(value-cpuResult[index]!))
      }
    }
    const diagnostics=universeDiagnostics(gpuResult,solver.params)
    const passes=nonfinite===0&&maxError<.002
    const name=solver.gpu.adapter.name
    display('verification-result',
      (passes?'PASS':'FAIL')+' · CPU vs actual WebGPU one-step delta: '+
      maxError.toExponential(3)+' (tolerance 2.0e-3), nonfinite: '+nonfinite+
      '. Mass: '+diagnostics.mass.toFixed(3)+
      '; total mechanical energy: '+diagnostics.total.toFixed(4)+
      '. Adapter: '+name+
      '. Software-adapter rejection is heuristic; this run is not a cross-device hardware certification.')
    solver.reset(initial)
  }catch(err){display('verification-result','Verification failed: '+String(err))}
  finally{verifying=false;verifyButton.disabled=false}
}
population.addEventListener('change',()=>{void begin(Number(population.value))})
for(const slider of [gravity,softening,zoom])slider.addEventListener('input',setParameters)
pauseButton.addEventListener('click',()=>{
  paused=!paused
  pauseButton.textContent=paused?'Resume':'Pause'
})
byId<HTMLButtonElement>('reset').addEventListener('click',()=>{
  if(solver)solver.reset(initial)
})
verifyButton.addEventListener('click',()=>{void verify()})
let dragging=false,previousX=0,previousY=0
canvas.addEventListener('pointerdown',event=>{
  dragging=true;previousX=event.clientX;previousY=event.clientY
  canvas.setPointerCapture(event.pointerId)
})
canvas.addEventListener('pointermove',event=>{
  if(!dragging||!viewer)return
  viewer.yaw+=(event.clientX-previousX)*.006
  viewer.pitch=Math.max(-1.45,Math.min(1.45,viewer.pitch+(event.clientY-previousY)*.006))
  previousX=event.clientX;previousY=event.clientY
})
canvas.addEventListener('pointerup',()=>{dragging=false})
canvas.addEventListener('pointercancel',()=>{dragging=false})
canvas.addEventListener('wheel',event=>{
  event.preventDefault()
  zoom.value=String(Math.max(1,Math.min(6,Number(zoom.value)+Math.sign(event.deltaY)*.18)))
  setParameters()
},{passive:false})
window.addEventListener('pagehide',()=>{cancelAnimationFrame(raf);viewer?.dispose();solver?.dispose()})
gravity.value=String(DEFAULT_G);softening.value=String(DEFAULT_SOFTENING)
void begin(256)

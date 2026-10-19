import {CataclysmSphGpu} from './solver.ts'
import {CataclysmRenderer} from './renderer.ts'
import {stepSphCpu,validateSph,type SphSettings} from './model.ts'
import {makeCataclysmPreset,cataclysmSettings,
 type CataclysmPreset} from './presets.ts'
const el=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T
const canvas=el<HTMLCanvasElement>('scene')
const preset=el<HTMLSelectElement>('preset')
const gravity=el<HTMLInputElement>('gravity')
const stiffness=el<HTMLInputElement>('stiffness')
const viscosity=el<HTMLInputElement>('viscosity')
const zoom=el<HTMLInputElement>('zoom')
const pause=el<HTMLButtonElement>('pause')
const verify=el<HTMLButtonElement>('verify')
const error=el<HTMLElement>('error')
const print=(id:string,text:string)=>{el<HTMLElement>(id).textContent=text}
const isLarge=()=>preset.value==='dense-4096'||preset.value==='stress-18432'
const parameters=():SphSettings=>{
 const base=cataclysmSettings(Number(gravity.value),
  Number(stiffness.value),Number(viscosity.value))
 return isLarge()?{...base,dt:.001,restDensity:25}:base
}
let simulation:CataclysmSphGpu|null=null,viewer:CataclysmRenderer|null=null
let frameHandle=0,epoch=0,paused=false,checking=false,frames=0,lastFps=performance.now()
function displayControls(){
 print('gravity-value',Number(gravity.value).toFixed(1))
 print('stiffness-value',Number(stiffness.value).toFixed(1))
 print('viscosity-value',Number(viscosity.value).toFixed(2))
 print('zoom-value',Number(zoom.value).toFixed(1))
 if(viewer)viewer.distance=Number(zoom.value)
}
function fail(cause:unknown){
 error.hidden=false
 error.textContent='WebGPU fluid simulation stopped: '+String(
  cause instanceof Error?cause.message:cause)
 print('gpu-status','GPU UNAVAILABLE')
}
function draw(now:number){
 if(!simulation||!viewer)return
 try{
  if(!paused&&!checking&&simulation.stepCount<1400)simulation.step()
  if(simulation.stepCount===1400){paused=true;pause.textContent='Restart'}
  viewer.render()
  frames++
  print('step-count',simulation.stepCount.toLocaleString())
  print('time-count',(simulation.stepCount*simulation.settings.dt).toFixed(2))
  if(now-lastFps>1000){
   print('fps',String(Math.round(frames*1000/(now-lastFps))))
   lastFps=now;frames=0
  }
  frameHandle=requestAnimationFrame(draw)
 }catch(e){fail(e)}
}
async function start(){
 const token=++epoch
 cancelAnimationFrame(frameHandle)
 viewer?.dispose();simulation?.dispose()
 viewer=null;simulation=null;paused=false;checking=false
 error.hidden=true;pause.textContent='Pause'
 print('gpu-status','COMPILING GPU HYDRODYNAMICS')
 print('verification',isLarge()?
  'Large GPU-only run: Verify checks mass conservation, tank containment and finite values, not an O(N²) CPU comparison.':
  'Click Verify to compare a live SPH step with the independent CPU numerical oracle.')
 displayControls()
 try{
  const initial=makeCataclysmPreset(preset.value as CataclysmPreset)
  validateSph(initial,parameters(),isLarge()?131_072:1024)
  const engine=await CataclysmSphGpu.create(initial,parameters())
  if(token!==epoch){engine.dispose();return}
  const scene=await CataclysmRenderer.create(engine,canvas)
  if(token!==epoch){scene.dispose();engine.dispose();return}
  simulation=engine;viewer=scene
  viewer.distance=Number(zoom.value)
  print('particle-count',engine.count.toLocaleString())
  print('adapter',engine.gpu.adapter.name)
  print('memory',(engine.gpu.bytesAllocated()/1024).toFixed(1))
  print('gpu-status','WEBGPU FLUID COMPUTE ACTIVE')
  frames=0;lastFps=performance.now()
  frameHandle=requestAnimationFrame(draw)
 }catch(e){if(token===epoch)fail(e)}
}
async function verifyStep(){
 if(!simulation||checking)return
 checking=true;paused=true;pause.textContent='Resume';verify.disabled=true
 try{
  const before=await simulation.read()
  const expected=isLarge()?null:stepSphCpu(before,parameters())
  simulation.step()
  await simulation.flush()
  const after=await simulation.read()
  let invalid=0
  if(expected!==null){
   let maximum=0
   for(const axis of ['positions','velocities'] as const)
    for(let i=0;i<expected[axis].length;i++){
     const observed=after[axis][i]??Number.NaN
     if(!Number.isFinite(observed))invalid++
     maximum=Math.max(maximum,Math.abs(observed-(expected[axis][i]??Number.NaN)))
    }
   const good=maximum<.015&&invalid===0
   print('verification',(good?'PASS':'FAIL')+' · GPU vs independent CPU next step: '+
    maximum.toExponential(3)+' maximum component error; '+invalid+
    ' nonfinite components. Adapter: '+simulation.gpu.adapter.name+
    '. Not a tsunami validation.')
  }else{
   const limits=simulation.settings
   let initialMass=0,actualMass=0,boundaryViolations=0,maxSpeed=0
   for(let i=0;i<after.positions.length;i+=4){
    initialMass+=before.positions[i+3]??0
    actualMass+=after.positions[i+3]??0
    let speedSquared=0
    for(let axis=0;axis<3;axis++){
     const pos=after.positions[i+axis]??Number.NaN
     const vel=after.velocities[i+axis]??Number.NaN
     if(!Number.isFinite(pos)||!Number.isFinite(vel))invalid++
     if(pos<limits.min[axis]!-1e-4||pos>limits.max[axis]!+1e-4)
      boundaryViolations++
     speedSquared+=vel*vel
    }
    maxSpeed=Math.max(maxSpeed,Math.sqrt(speedSquared))
   }
   const massError=Math.abs(initialMass-actualMass)
   const good=invalid===0&&boundaryViolations===0&&massError<=1e-5&&maxSpeed<250
   print('verification',(good?'PASS':'FAIL')+
    ' · GPU-only invariants for '+simulation.count.toLocaleString()+
    ' particles: mass error '+massError.toExponential(2)+
    ', nonfinite '+invalid+', out-of-tank coordinates '+boundaryViolations+
    ', peak speed '+maxSpeed.toFixed(3)+
    '. No all-pairs CPU equivalence implied.')
  }
 }catch(e){print('verification','Verification failed: '+String(e))}
 finally{checking=false;verify.disabled=false}
}
preset.addEventListener('change',()=>{
  stiffness.value=isLarge()?'2.5':'12'
  viscosity.value=isLarge()?'.025':'.12'
  void start()
})
for(const slider of [gravity,stiffness,viscosity]){
 slider.addEventListener('input',displayControls)
 slider.addEventListener('change',()=>{void start()})
}
zoom.addEventListener('input',displayControls)
pause.addEventListener('click',()=>{
 if(simulation?.stepCount===1400){void start();return}
 paused=!paused;pause.textContent=paused?'Resume':'Pause'
})
el<HTMLButtonElement>('reset').addEventListener('click',()=>{void start()})
verify.addEventListener('click',()=>{void verifyStep()})
let moving=false,oldX=0,oldY=0
canvas.addEventListener('pointerdown',e=>{
 if(!viewer)return
 moving=true;oldX=e.clientX;oldY=e.clientY
 canvas.setPointerCapture(e.pointerId)
})
canvas.addEventListener('pointermove',e=>{
 if(!moving||!viewer)return
 viewer.yaw+=(e.clientX-oldX)*.006
 viewer.pitch=Math.max(-1.2,Math.min(1.2,viewer.pitch+(e.clientY-oldY)*.006))
 oldX=e.clientX;oldY=e.clientY
})
canvas.addEventListener('pointerup',()=>{moving=false})
canvas.addEventListener('pointercancel',()=>{moving=false})
canvas.addEventListener('wheel',e=>{
 e.preventDefault()
 zoom.value=String(Math.max(9,Math.min(34,Number(zoom.value)+Math.sign(e.deltaY)*.6)))
 displayControls()
},{passive:false})
window.addEventListener('pagehide',()=>{
 cancelAnimationFrame(frameHandle);viewer?.dispose();simulation?.dispose()
})
void start()

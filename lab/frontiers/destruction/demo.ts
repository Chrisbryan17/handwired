import {SphereContactGpu} from '../../../src/runtime/frontiers-contacts-gpu.ts'
import {stepSphereContactsCpu,rigidBodyDiagnostics,
 type SphereContactSettings,type SphereBodies} from '../../../src/runtime/frontiers-contacts.ts'
import {makeImpactScenario,impactSpatialGrid,impactDefaults,
 type ImpactPreset} from './model.ts'
import {ContactSceneRenderer} from './renderer.ts'
const byId=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T
const canvas=byId<HTMLCanvasElement>('scene')
const preset=byId<HTMLSelectElement>('preset')
const gravity=byId<HTMLInputElement>('gravity')
const restitution=byId<HTMLInputElement>('restitution')
const friction=byId<HTMLInputElement>('friction')
const zoom=byId<HTMLInputElement>('camera-distance')
const pause=byId<HTMLButtonElement>('pause')
const verify=byId<HTMLButtonElement>('verify')
const error=byId<HTMLElement>('error')
const value=(name:string,text:string)=>{byId<HTMLElement>(name).textContent=text}
let engine:SphereContactGpu|null=null,view:ContactSceneRenderer|null=null
let epoch=0,animation=0,paused=false,checking=false,frames=0,last=performance.now()
function controls():SphereContactSettings{
 return {...impactDefaults,gravity:[0,-Number(gravity.value),0],
   restitution:Number(restitution.value),friction:Number(friction.value)}
}
function labels(){
 value('gravity-value',Number(gravity.value).toFixed(1))
 value('restitution-value',Number(restitution.value).toFixed(2))
 value('friction-value',Number(friction.value).toFixed(2))
 value('zoom-value',Number(zoom.value).toFixed(1))
 if(view)view.distance=Number(zoom.value)
}
function fail(cause:unknown){
 error.hidden=false
 error.textContent='Real GPU simulation interrupted: '+String(cause instanceof Error?cause.message:cause)
 value('gpu-status','GPU ERROR')
}
function tick(now:number){
 if(!engine||!view)return
 try{
  if(!paused&&!checking&&engine.steps<720)engine.step()
  if(engine.steps===720){paused=true;pause.textContent='Restart simulation'}
  view.render()
  frames++
  value('step-count',engine.steps.toLocaleString())
  if(now-last>=950){
   value('fps',String(Math.round(frames*1000/(now-last))))
   frames=0;last=now
  }
  animation=requestAnimationFrame(tick)
 }catch(e){fail(e)}
}
async function start(){
 const current=++epoch
 cancelAnimationFrame(animation)
 view?.dispose();engine?.dispose();view=null;engine=null
 paused=false;checking=false;pause.textContent='Pause'
 error.hidden=true
 value('gpu-status','COMPILING GPU CONTACTS')
 value('verification','Select the verification action to independently check this GPU state.')
 labels()
 const initial=makeImpactScenario(preset.value as ImpactPreset)
 try{
  const next=await SphereContactGpu.create(initial,impactSpatialGrid,controls())
  if(current!==epoch){next.dispose();return}
  const renderer=await ContactSceneRenderer.create(next,canvas,initial)
  if(current!==epoch){renderer.dispose();next.dispose();return}
  engine=next;view=renderer
  view.distance=Number(zoom.value)
  view.target=[0,0,0]
  value('body-count',String(next.count))
  value('adapter',next.grid.gpu.adapter.name)
  value('memory',(next.grid.gpu.bytesAllocated()/1024).toFixed(1))
  value('gpu-status','PHYSICS COMPUTE ACTIVE')
  frames=0;last=performance.now()
  animation=requestAnimationFrame(tick)
 }catch(e){if(current===epoch)fail(e)}
}
async function check(){
 if(!engine||checking)return
 checking=true;paused=true;pause.textContent='Resume';verify.disabled=true
 try{
  if(engine.steps===0){engine.step();await engine.flush()}
  const before=await engine.readState()
  const oracle=stepSphereContactsCpu(before,controls())
  engine.step()
  await engine.flush()
  const actual=await engine.readState()
  let maximum=0,nonfinite=0
  for(const channel of ['positions','velocities'] as const){
   for(let i=0;i<oracle.state[channel].length;i++){
    const observed=actual[channel][i]!
    if(!Number.isFinite(observed))nonfinite++
    maximum=Math.max(maximum,Math.abs(observed-oracle.state[channel][i]!))
   }
  }
  const diagnostics=rigidBodyDiagnostics(actual)
  const ok=maximum<=.002&&nonfinite===0
  value('verification',(ok?'PASS':'FAIL')+' · synchronized GPU vs independent CPU: '+
    maximum.toExponential(3)+' maximum absolute state error, '+nonfinite+
    ' invalid values. Kinetic energy: '+diagnostics.kinetic.toFixed(3)+
    ' model units; CPU pairs resolved: '+oracle.contactsProcessed+
    '. Adapter: '+engine.grid.gpu.adapter.name+
    '. Limited to one current-state Jacobi contact update; not fracture validation.')
 }catch(e){value('verification','Verification failed: '+String(e))}
 finally{checking=false;verify.disabled=false}
}
preset.addEventListener('change',()=>{void start()})
for(const input of [gravity,restitution,friction]){
 input.addEventListener('input',labels)
 input.addEventListener('change',()=>{void start()})
}
zoom.addEventListener('input',labels)
pause.addEventListener('click',()=>{
 if(engine?.steps===720){void start();return}
 paused=!paused;pause.textContent=paused?'Resume':'Pause'
})
byId<HTMLButtonElement>('restart').addEventListener('click',()=>{void start()})
verify.addEventListener('click',()=>{void check()})
let touching=false,x=0,y=0
canvas.addEventListener('pointerdown',e=>{
 if(!view)return
 touching=true;x=e.clientX;y=e.clientY;canvas.setPointerCapture(e.pointerId)
})
canvas.addEventListener('pointermove',e=>{
 if(!touching||!view)return
 view.yaw+=(e.clientX-x)*.006
 view.pitch=Math.max(-1.15,Math.min(1.15,view.pitch+(e.clientY-y)*.006))
 x=e.clientX;y=e.clientY
})
canvas.addEventListener('pointerup',()=>{touching=false})
canvas.addEventListener('pointercancel',()=>{touching=false})
canvas.addEventListener('wheel',e=>{
 e.preventDefault()
 zoom.value=String(Math.min(37,Math.max(10,Number(zoom.value)+Math.sign(e.deltaY)*.7)))
 labels()
},{passive:false})
window.addEventListener('pagehide',()=>{
 cancelAnimationFrame(animation);view?.dispose();engine?.dispose()
})
void start()

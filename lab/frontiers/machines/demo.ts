import {BatchedLegsGpu} from './gpu.ts'
import {MachinesRenderer} from './renderer.ts'
import {makeLegPopulation,stepLegCpu,legDefaults,
 type PlanarLegParameters} from './model.ts'
const byId=<T extends HTMLElement>(key:string)=>document.getElementById(key) as T
const canvas=byId<HTMLCanvasElement>('scene')
const population=byId<HTMLSelectElement>('population')
const hip=byId<HTMLInputElement>('target-hip')
const knee=byId<HTMLInputElement>('target-knee')
const kp=byId<HTMLInputElement>('kp')
const kd=byId<HTMLInputElement>('kd')
const gravity=byId<HTMLInputElement>('gravity')
const zoom=byId<HTMLInputElement>('zoom')
const pause=byId<HTMLButtonElement>('pause')
const verify=byId<HTMLButtonElement>('verify')
const error=byId<HTMLElement>('error')
const update=(name:string,v:string)=>{byId<HTMLElement>(name).textContent=v}
const parameters=():PlanarLegParameters=>({
 ...legDefaults,gravity:Number(gravity.value),
 targetHip:Number(hip.value),targetKnee:Number(knee.value),
 kp:Number(kp.value),kd:Number(kd.value)
})
let simulation:BatchedLegsGpu|null=null,scene:MachinesRenderer|null=null
let animation=0,epoch=0,paused=false,verifying=false
let frames=0,last=performance.now()
function labels(){
 update('target-hip-value',Number(hip.value).toFixed(2))
 update('target-knee-value',Number(knee.value).toFixed(2))
 update('kp-value',Number(kp.value).toFixed(1))
 update('kd-value',Number(kd.value).toFixed(1))
 update('gravity-value',Number(gravity.value).toFixed(1))
 update('zoom-value',String(Math.round(Number(zoom.value))))
 if(scene)scene.distance=Number(zoom.value)
}
function fail(e:unknown){
 error.hidden=false
 error.textContent='Articulated GPU research error: '+String(
  e instanceof Error?e.message:e)
 update('gpu-status','GPU ERROR')
}
function tick(now:number){
 if(!simulation||!scene)return
 try{
  if(!paused&&!verifying&&simulation.steps<2400)simulation.step()
  if(simulation.steps===2400){paused=true;pause.textContent='Restart'}
  scene.render()
  frames++
  update('steps',simulation.steps.toLocaleString())
  update('model-time',(simulation.steps*simulation.parameters.dt).toFixed(2))
  if(now-last>=1000){
   update('fps',String(Math.round(frames*1000/(now-last))))
   frames=0;last=now
  }
  animation=requestAnimationFrame(tick)
 }catch(e){fail(e)}
}
async function start(){
 const version=++epoch
 cancelAnimationFrame(animation)
 scene?.dispose();simulation?.dispose()
 scene=null;simulation=null
 paused=false;verifying=false;pause.textContent='Pause';error.hidden=true
 update('gpu-status','COMPILING COUPLED INERTIA')
 update('verification','Check a physically updated GPU step against the independent CPU leg oracle.')
 labels()
 try{
  const count=Number(population.value)
  const state=makeLegPopulation(count,7831)
  const engine=await BatchedLegsGpu.create(state,parameters())
  if(version!==epoch){engine.dispose();return}
  const viewer=await MachinesRenderer.create(engine,canvas)
  if(version!==epoch){viewer.dispose();engine.dispose();return}
  simulation=engine;scene=viewer
  zoom.value=String(Math.round(viewer.distance))
  labels()
  update('count',engine.count.toLocaleString())
  update('degrees',(engine.count*2).toLocaleString())
  update('adapter',engine.gpu.adapter.name)
  update('memory',(engine.gpu.bytesAllocated()/1024).toFixed(1))
  update('gpu-status','REAL 2DOF GPU DYNAMICS ACTIVE')
  frames=0;last=performance.now()
  animation=requestAnimationFrame(tick)
 }catch(e){if(epoch===version)fail(e)}
}
async function check(){
 if(!simulation||verifying)return
 verifying=true;paused=true;pause.textContent='Resume';verify.disabled=true
 try{
  const before=await simulation.read()
  const predicted=stepLegCpu(before,simulation.parameters)
  simulation.step()
  await simulation.flush()
  const result=await simulation.read()
  let maximum=0,invalid=0,mean=0
  for(let i=0;i<result.joints.length;i++){
   const v=result.joints[i]??Number.NaN
   if(!Number.isFinite(v))invalid++
   const delta=Math.abs(v-(predicted.joints[i]??Number.NaN))
   maximum=Math.max(maximum,delta)
   mean+=delta
  }
  const good=maximum<=.018&&invalid===0
  update('verification',(good?'PASS':'FAIL')+
   ' · GPU vs independent CPU full joint arrays: '+
   maximum.toExponential(3)+' max error, '+
   (mean/result.joints.length).toExponential(2)+' mean, '+
   invalid+' invalid components across '+
   simulation.count.toLocaleString()+' articulated legs. '+
   'Not trained locomotion or free-base physics.')
 }catch(e){update('verification','Verification failed: '+String(e))}
 finally{verifying=false;verify.disabled=false}
}
population.addEventListener('change',()=>{void start()})
for(const slider of [hip,knee,kp,kd,gravity]){
 slider.addEventListener('input',labels)
 slider.addEventListener('change',()=>{void start()})
}
zoom.addEventListener('input',labels)
pause.addEventListener('click',()=>{
 if(simulation?.steps===2400){void start();return}
 paused=!paused;pause.textContent=paused?'Resume':'Pause'
})
byId<HTMLButtonElement>('reset').addEventListener('click',()=>{void start()})
verify.addEventListener('click',()=>{void check()})
let dragging=false,x=0,y=0
canvas.addEventListener('pointerdown',e=>{
 if(!scene)return
 dragging=true;x=e.clientX;y=e.clientY;canvas.setPointerCapture(e.pointerId)
})
canvas.addEventListener('pointermove',e=>{
 if(!dragging||!scene)return
 scene.yaw+=(e.clientX-x)*.006
 scene.pitch=Math.max(-1.2,Math.min(1.2,scene.pitch+(e.clientY-y)*.006))
 x=e.clientX;y=e.clientY
})
canvas.addEventListener('pointerup',()=>{dragging=false})
canvas.addEventListener('pointercancel',()=>{dragging=false})
canvas.addEventListener('wheel',e=>{
 e.preventDefault()
 zoom.value=String(Math.max(10,Math.min(330,Number(zoom.value)+Math.sign(e.deltaY)*3)))
 labels()
},{passive:false})
window.addEventListener('pagehide',()=>{
 cancelAnimationFrame(animation);scene?.dispose();simulation?.dispose()
})
void start()

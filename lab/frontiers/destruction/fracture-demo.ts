import {BondedStructureGpu} from './fracture-gpu.ts'
import {FractureRenderer} from './fracture-renderer.ts'
import {makeBondedBridge,stepBondedCpu,structuralDiagnostics,
 type BondedBody,type BondedSettings} from './fracture-model.ts'
const id=<T extends HTMLElement>(key:string)=>document.getElementById(key) as T
const canvas=id<HTMLCanvasElement>('scene')
const structure=id<HTMLSelectElement>('structure')
const gravity=id<HTMLInputElement>('gravity')
const strength=id<HTMLInputElement>('strength')
const zoom=id<HTMLInputElement>('camera-distance')
const pause=id<HTMLButtonElement>('pause')
const verify=id<HTMLButtonElement>('verify')
const failure=id<HTMLElement>('error')
const report=(name:string,text:string)=>{id<HTMLElement>(name).textContent=text}
let solver:BondedStructureGpu|null=null,renderer:FractureRenderer|null=null
let run=0,animation=0,paused=false,checking=false,frames=0,last=performance.now()
let state:BondedBody|null=null
let columns=14,levels=4
const parameters=():BondedSettings=>({dt:1/120,gravity:[0,-Number(gravity.value),0]})
function labels(){
 report('gravity-value',Number(gravity.value).toFixed(2))
 report('strength-value',Number(strength.value).toFixed(2)+'×')
 report('zoom-value',Number(zoom.value).toFixed(1))
 if(renderer)renderer.distance=Number(zoom.value)
}
function make():BondedBody{
 columns=structure.value==='dense'?24:structure.value==='compact'?8:14
 levels=structure.value==='dense'?5:structure.value==='compact'?3:4
 const data=makeBondedBridge(columns,levels,.65)
 const strengthFactor=Number(strength.value)
 for(let i=0;i<data.bonds.length;i+=8){
  data.bonds[i+3]=Math.fround(data.bonds[i+3]!*strengthFactor)
  data.bonds[i+4]=Math.fround(data.bonds[i+4]!*strengthFactor)
 }
 return data
}
function fail(e:unknown){
 failure.hidden=false
 failure.textContent='GPU fracture stopped: '+String(e instanceof Error?e.message:e)
 report('gpu-status','GPU UNAVAILABLE')
}
function tick(now:number){
 if(!solver||!renderer)return
 try{
  if(!paused&&!checking&&solver.stepCount<1800)solver.step()
  if(solver.stepCount===1800){paused=true;pause.textContent='Rebuild'}
  renderer.render()
  frames++
  report('step-count',solver.stepCount.toLocaleString())
  if(now-last>1000){
   report('fps',String(Math.round(frames*1000/(now-last))))
   frames=0;last=now
  }
  animation=requestAnimationFrame(tick)
 }catch(e){fail(e)}
}
async function start(){
 const active=++run
 cancelAnimationFrame(animation);renderer?.dispose();solver?.dispose()
 renderer=null;solver=null;paused=false;checking=false
 failure.hidden=true;pause.textContent='Pause'
 report('gpu-status','COMPILING STRUCTURE')
 report('verification','Click Verify to perform independent CPU/GPU comparison.')
 labels()
 state=make()
 try{
  const engine=await BondedStructureGpu.create(state,parameters())
  if(active!==run){engine.dispose();return}
  const viewer=await FractureRenderer.create(engine,canvas)
  if(active!==run){viewer.dispose();engine.dispose();return}
  solver=engine;renderer=viewer
  renderer.distance=Number(zoom.value)
  report('gpu-status','GPU FRACTURE ACTIVE')
  report('adapter',engine.gpu.adapter.name)
  report('node-count',String(engine.nodeCount))
  report('bond-count',String(engine.bondCount))
  report('memory',(engine.gpu.bytesAllocated()/1024).toFixed(1))
  frames=0;last=performance.now()
  animation=requestAnimationFrame(tick)
 }catch(e){if(active===run)fail(e)}
}
function strike(column:number){
 if(!solver)return
 const j=Math.floor(levels/2)*2
 const node=column*levels*2+j
 try{
  solver.applyImpulse(node,[0,-17,3.5])
  report('verification','User impulse applied to node '+node+' on the GPU. Run Verify to compare the next physical step.')
 }catch(e){fail(e)}
}
async function check(){
 if(!solver||checking)return
 checking=true;paused=true;pause.textContent='Resume';verify.disabled=true
 try{
  if(solver.stepCount===0){solver.step();await solver.flush()}
  const before=await solver.read()
  const expected=stepBondedCpu(before,parameters())
  solver.step()
  await solver.flush()
  const actual=await solver.read()
  let error=0,invalid=0,brokenMismatch=0
  for(const name of ['positions','velocities'] as const)
   for(let i=0;i<actual[name].length;i++){
    const v=actual[name][i]!
    if(!Number.isFinite(v))invalid++
    error=Math.max(error,Math.abs(v-expected.body[name][i]!))
   }
  for(let i=0;i<actual.bonds.length;i+=8){
   const damage=actual.bonds[i+6]!
   error=Math.max(error,Math.abs(damage-expected.body.bonds[i+6]!))
   if(actual.bonds[i+7]!==expected.body.bonds[i+7])brokenMismatch++
  }
  const diagnostic=structuralDiagnostics(actual)
  const ok=error<.006&&invalid===0&&brokenMismatch===0
  report('verification',(ok?'PASS':'FAIL')+' · maximum GPU/CPU error '+
   error.toExponential(3)+'; broken links '+diagnostic.broken+
   '; new damage events '+expected.diagnostics.events+
   '; invalid values '+invalid+'; bond mismatches '+brokenMismatch+
   '. Adapter '+solver.gpu.adapter.name+
   '. Not a continuum/FEM engineering prediction.')
 }catch(e){report('verification','Verification failed: '+String(e))}
 finally{checking=false;verify.disabled=false}
}
for(const el of [structure,gravity,strength]){
 el.addEventListener('change',()=>{void start()})
}
for(const el of [gravity,strength])el.addEventListener('input',labels)
zoom.addEventListener('input',labels)
id<HTMLButtonElement>('strike-left').addEventListener('click',()=>strike(Math.floor(columns*.25)))
id<HTMLButtonElement>('strike-center').addEventListener('click',()=>strike(Math.floor(columns*.5)))
id<HTMLButtonElement>('strike-right').addEventListener('click',()=>strike(Math.floor(columns*.75)))
pause.addEventListener('click',()=>{
 if(solver?.stepCount===1800){void start();return}
 paused=!paused;pause.textContent=paused?'Resume':'Pause'
})
id<HTMLButtonElement>('restart').addEventListener('click',()=>{void start()})
verify.addEventListener('click',()=>{void check()})
let dragging=false,x=0,y=0
canvas.addEventListener('pointerdown',e=>{
 if(!renderer)return
 dragging=true;x=e.clientX;y=e.clientY;canvas.setPointerCapture(e.pointerId)
})
canvas.addEventListener('pointermove',e=>{
 if(!dragging||!renderer)return
 renderer.yaw+=(e.clientX-x)*.006
 renderer.pitch=Math.max(-1.15,Math.min(1.15,renderer.pitch+(e.clientY-y)*.006))
 x=e.clientX;y=e.clientY
})
canvas.addEventListener('pointerup',()=>{dragging=false})
canvas.addEventListener('pointercancel',()=>{dragging=false})
window.addEventListener('pagehide',()=>{
 cancelAnimationFrame(animation);renderer?.dispose();solver?.dispose()
})
void start()

import {genesisDefaults,makeGenesis,stepGenesisCpu,genesisStatistics,
  type GenesisField,type GenesisParams} from './model.ts'
import {GenesisGpuSolver} from './solver.ts'
import {GenesisRenderer} from './renderer.ts'
const element=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T
const canvas=element<HTMLCanvasElement>('field')
const dimensions=element<HTMLSelectElement>('dimensions')
const presets=element<HTMLSelectElement>('presets')
const feed=element<HTMLInputElement>('feed')
const kill=element<HTMLInputElement>('kill')
const speed=element<HTMLInputElement>('speed')
const palette=element<HTMLSelectElement>('palette')
const brushMode=element<HTMLSelectElement>('brush-mode')
const error=element<HTMLElement>('error')
const pause=element<HTMLButtonElement>('pause')
const verify=element<HTMLButtonElement>('verify')
const setText=(id:string,text:string)=>{element<HTMLElement>(id).textContent=text}
const presetsMap:Record<string,{feed:number,kill:number}>={
 coral:{feed:.035,kill:.062},spots:{feed:.030,kill:.062},
 stripes:{feed:.026,kill:.051},waves:{feed:.022,kill:.049},
}
let solver:GenesisGpuSolver|null=null,viewer:GenesisRenderer|null=null
let initial:GenesisField=makeGenesis(192,192)
let raf=0,run=0,paused=false,checking=false,fpsFrames=0,fpsStart=performance.now()
let painting=false,erase=false
const currentParams=():GenesisParams=>({
 feed:Number(feed.value),kill:Number(kill.value),
 diffuseU:genesisDefaults.diffuseU,diffuseV:genesisDefaults.diffuseV,
})
const writeParameters=()=>{
 setText('feed-value',Number(feed.value).toFixed(3))
 setText('kill-value',Number(kill.value).toFixed(3))
 setText('speed-value',speed.value)
 if(solver)solver.setParameters(currentParams())
 if(viewer)viewer.palette=Number(palette.value)
}
function reportFailure(err:unknown){
 error.hidden=false
 error.textContent='GPU simulation interrupted: '+String(err instanceof Error?err.message:err)
 setText('gpu','GPU COMPUTE UNAVAILABLE')
}
function paint(event:PointerEvent){
 if(!painting||!solver)return
 const rect=canvas.getBoundingClientRect()
 const x=Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))
 const y=Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))
 solver.setBrush({x,y,mode:erase?2:1,radius:Math.max(3,solver.width*.025)})
}
function stopPaint(){
 painting=false
 if(solver)solver.setBrush({x:.5,y:.5,mode:0,radius:8})
}
function frame(time:number){
 if(!solver||!viewer)return
 try{
  const advance=!paused&&!checking
  if(advance){
   const count=Number(speed.value)
   if(count>1)solver.step(count-1)
  }
  viewer.render(advance)
  fpsFrames++
  setText('steps',solver.frame.toLocaleString())
  if(time-fpsStart>1000){
    setText('fps',String(Math.round(fpsFrames*1000/(time-fpsStart))))
    fpsFrames=0;fpsStart=time
  }
  raf=requestAnimationFrame(frame)
 }catch(err){reportFailure(err)}
}
async function start(size:number){
 const epoch=++run
 cancelAnimationFrame(raf);viewer?.dispose();solver?.dispose()
 solver=null;viewer=null;error.hidden=true;painting=false
 paused=false;checking=false;pause.textContent='Pause'
 setText('gpu','INITIALIZING WEBGPU')
 initial=makeGenesis(size,size,17)
 try{
  const engine=await GenesisGpuSolver.create(initial,currentParams())
  if(epoch!==run){engine.dispose();return}
  const renderer=await GenesisRenderer.create(engine,canvas)
  if(epoch!==run){renderer.dispose();engine.dispose();return}
  solver=engine;viewer=renderer;writeParameters()
  setText('gpu','WEBGPU CHEMISTRY ACTIVE')
  setText('adapter',engine.gpu.adapter.name)
  setText('cells',(size*size).toLocaleString())
  setText('dimensions-value',size+' × '+size)
  setText('size',((size*size*2*4)/1024).toFixed(1))
  setText('memory',(engine.gpu.bytesAllocated()/1024).toFixed(1))
  setText('verification','A synchronized numerical comparison is available for this device.')
  fpsFrames=0;fpsStart=performance.now();raf=requestAnimationFrame(frame)
 }catch(err){if(epoch===run)reportFailure(err)}
}
async function verifyOneStep(){
 if(!solver||checking)return
 checking=true;paused=true;pause.textContent='Resume';verify.disabled=true
 setText('verification','Synchronizing WebGPU and computing the independent Gray–Scott oracle…')
 try{
  solver.setBrush({x:.5,y:.5,mode:0,radius:8})
  solver.reset(initial)
  solver.step()
  const gpu=await solver.read()
  const cpu=stepGenesisCpu(initial,solver.params)
  let maxError=0,invalid=0
  for(let i=0;i<gpu.state.length;i++){
   if(!Number.isFinite(gpu.state[i]))invalid++
   maxError=Math.max(maxError,Math.abs(gpu.state[i]!-cpu.state[i]!))
  }
  const stats=genesisStatistics(gpu)
  const passed=invalid===0&&maxError<.0003
  setText('verification',(passed?'PASS':'FAIL')+' · max concentration difference '+
   maxError.toExponential(3)+' / 3.0e-4, invalid '+invalid+
   ', mean V '+stats.meanV.toFixed(5)+', cells '+solver.numCells.toLocaleString()+
   ' · '+solver.gpu.adapter.name+
   '. One-step agreement does not establish long-running accuracy.')
  solver.reset(initial)
 }catch(err){setText('verification','Verification failed: '+String(err))}
 finally{checking=false;verify.disabled=false}
}
dimensions.addEventListener('change',()=>{void start(Number(dimensions.value))})
presets.addEventListener('change',()=>{
 const p=presetsMap[presets.value]
 if(!p)return
 feed.value=String(p.feed);kill.value=String(p.kill);writeParameters()
})
feed.addEventListener('input',writeParameters)
kill.addEventListener('input',writeParameters)
speed.addEventListener('input',writeParameters)
palette.addEventListener('change',writeParameters)
pause.addEventListener('click',()=>{paused=!paused;pause.textContent=paused?'Resume':'Pause'})
element<HTMLButtonElement>('reset').addEventListener('click',()=>{
 stopPaint()
 if(solver)solver.reset(initial)
})
verify.addEventListener('click',()=>{void verifyOneStep()})
canvas.addEventListener('contextmenu',event=>event.preventDefault())
canvas.addEventListener('pointerdown',event=>{
 if(!solver)return
 painting=true;erase=event.button===2||event.shiftKey||brushMode.value==='2'
 canvas.setPointerCapture(event.pointerId);paint(event)
})
canvas.addEventListener('pointermove',paint)
canvas.addEventListener('pointerup',stopPaint)
canvas.addEventListener('pointercancel',stopPaint)
window.addEventListener('pagehide',()=>{
 cancelAnimationFrame(raf);viewer?.dispose();solver?.dispose()
})
void start(192)

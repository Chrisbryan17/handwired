import {createWebGpuEvaluator,type WebGpuCircuitEvaluator} from '../../src/runtime/webgpu.ts'
import {ResidentMazeGpu} from './gpu-resident.ts'
import {buildMazeCircuit, MAZE_SOURCE_SIZE} from './model.ts'
import {beginSearch,circuitWave,generateMaze,mazeInputs,
  reconstructPath,referenceWave,type Maze,type SearchState} from './engine.ts'

const params=new URLSearchParams(location.search),smoke=params.get('smoke')==='1'
const {circuit,outputWires}=buildMazeCircuit()
const width=circuit.sourceSize+circuit.operations.length
const size=document.querySelector<HTMLSelectElement>('#size')!
const engine=document.querySelector<HTMLSelectElement>('#engine')!
const wantsFast=params.get('fast')==='1'
engine.value=smoke&&!wantsFast?'legacy':'fast'
const strategy=document.querySelector<HTMLSelectElement>('#strategy')!
const cpuCanvas=document.querySelector<HTMLCanvasElement>('#cpu')!
const gpuCanvas=document.querySelector<HTMLCanvasElement>('#gpu')!
const fastCanvas=document.querySelector<HTMLCanvasElement>('#gpu-fast')!
const cpuContext=cpuCanvas.getContext('2d')!,gpuContext=gpuCanvas.getContext('2d')!
const status=document.querySelector<HTMLElement>('#status')!
const put=(id:string,value:string)=>{document.getElementById(id)!.textContent=value}
let seed=1849,maze:Maze=generateMaze(),cpu:SearchState=beginSearch(maze),gpu:SearchState=beginSearch(maze)
let evaluator:WebGpuCircuitEvaluator|null=null
let resident:ResidentMazeGpu|null=null
let fastValidated=false
let frame=0,paused=false,restartRequested=true,stepRequested=false
let differences=0,maxError=0,checked=0
let cpuMs=0,gpuMs=0
const hues={wall:'#080c1d',unseen:'#151d39',seen:'#273d70',
  frontier:'#96f5d2',goal:'#ffb56e',start:'#fff7cf',path:'#e8cc6a'}

function canvasScale(canvas:HTMLCanvasElement,context:CanvasRenderingContext2D){
  const rect=canvas.getBoundingClientRect(),ratio=Math.min(2,window.devicePixelRatio||1)
  const ww=Math.max(1,Math.round(rect.width*ratio)),hh=Math.max(1,Math.round(rect.height*ratio))
  if(canvas.width!==ww||canvas.height!==hh){canvas.width=ww;canvas.height=hh}
  context.setTransform(ratio,0,0,ratio,0,0)
  return {width:rect.width,height:rect.height}
}
function draw(canvas:HTMLCanvasElement,ctx:CanvasRenderingContext2D,state:SearchState) {
  const bounds=canvasScale(canvas,ctx),cellW=bounds.width/maze.columns,cellH=bounds.height/maze.rows
  const gap=Math.max(.35,Math.min(1,cellW*.11))
  const path=state.visited[maze.goal]===1?new Set(reconstructPath(maze,state)):new Set<number>()
  ctx.fillStyle='#080c1d';ctx.fillRect(0,0,bounds.width,bounds.height)
  for(let i=0;i<maze.walls.length;i++){
    if(maze.walls[i]===1)continue
    const x=(i%maze.columns)*cellW,y=Math.floor(i/maze.columns)*cellH
    let color=state.frontier[i]===1?hues.frontier:state.visited[i]===1?hues.seen:hues.unseen
    if(path.has(i))color=hues.path
    if(i===maze.goal)color=hues.goal
    if(i===maze.start)color=hues.start
    ctx.fillStyle=color
    ctx.fillRect(x+gap,y+gap,cellW-2*gap,cellH-2*gap)
  }
}
async function restart(){
  evaluator?.destroy();evaluator=null;resident?.destroy();resident=null;fastValidated=false
  const cols=Number(size.value),rows=cols===31?21:cols===61?39:31
  maze=generateMaze(cols,rows,seed++)
  cpu=beginSearch(maze);gpu=beginSearch(maze)
  frame=0;differences=0;maxError=0;checked=0
  status.textContent='Compiling neural wavefront expansion…'
  document.body.classList.toggle('gpu-fast',engine.value==='fast')
  strategy.disabled=engine.value==='fast'
  if(engine.value==='fast'){
    const rect=fastCanvas.getBoundingClientRect(),scale=Math.min(2,window.devicePixelRatio||1)
    fastCanvas.width=Math.max(1,Math.round(rect.width*scale))
    fastCanvas.height=Math.max(1,Math.round(rect.height*scale))
    try{
      resident=await ResidentMazeGpu.create(fastCanvas,maze)
      status.textContent='Full GPU-resident wavefront: no per-step neighbor transfers or state readback.'
      put('gpu-info','GPU compute + GPU rendering · no per-step readback')
      put('wires','0 / step');put('errors','Waiting for full-state GPU/CPU verification')
      restartRequested=false
      draw(cpuCanvas,cpuContext,cpu)
      return
    }catch(error){
      if(smoke&&wantsFast)throw error
      engine.value='legacy';document.body.classList.remove('gpu-fast')
      status.textContent='GPU resident unavailable; falling back to legacy: '+String(error)
    }
  }
  evaluator=await createWebGpuEvaluator(circuit,{
    mode:smoke?'gpu':'auto',capacity:maze.walls.length,
    strategy:strategy.value as 'specialized'|'layers'|'monolithic'})
  status.textContent=evaluator.mode==='gpu'
    ? 'Live WebGPU expansion. Green cells are the current frontier.'
    : 'CPU fallback — real WebGPU unavailable: '+(evaluator.fallbackReason??'')
  put('gpu-info',evaluator.mode==='gpu'?'Actual WebGPU search circuit':'CPU fallback · no GPU claim')
  restartRequested=false
  draw(cpuCanvas,cpuContext,cpu);draw(gpuCanvas,gpuContext,gpu)
}
function compare(input:Float32Array,values:Float32Array){
  for(let i=0;i<maze.walls.length;i++){
    const expected=circuit.evaluateReference(input.subarray(i*MAZE_SOURCE_SIZE,(i+1)*MAZE_SOURCE_SIZE))
    for(let j=0;j<width;j++){
      const delta=Math.abs(expected[j]!-values[i*width+j]!)
      if(!Number.isFinite(delta)||delta>2e-5)differences++
      maxError=Math.max(maxError,delta);checked++
    }
  }
}
async function tick() {
  if(restartRequested||(evaluator===null&&resident===null))await restart()
  if(paused&&!stepRequested)return
  if(resident!==null){
    if(paused&&!stepRequested)return
    if(cpu.finished&&fastValidated&&!smoke)return
    const cpuStart=performance.now()
    if(!cpu.finished)cpu=referenceWave(maze,cpu)
    cpuMs=performance.now()-cpuStart
    const gpuStart=performance.now()
    const rect=fastCanvas.getBoundingClientRect(),ratio=Math.min(2,window.devicePixelRatio||1)
    const w=Math.max(1,Math.round(rect.width*ratio)),h=Math.max(1,Math.round(rect.height*ratio))
    if(fastCanvas.width!==w||fastCanvas.height!==h){fastCanvas.width=w;fastCanvas.height=h}
    resident.frame();gpuMs=performance.now()-gpuStart
    frame++;stepRequested=false
    draw(cpuCanvas,cpuContext,cpu)
    put('depth',String(frame))
    put('explored',cpu.visited.reduce((sum,x)=>sum+x,0).toLocaleString()+' (CPU)')
    put('cpu-info',cpuMs.toFixed(2)+'ms · exact CPU oracle')
    put('gpu-info',gpuMs.toFixed(2)+'ms CPU submission · no readback')
    const finish=smoke?frame>=8:cpu.finished
    if(finish&&!fastValidated){
      const result=await resident.read()
      let mismatches=0,maxError=0
      for(let i=0;i<maze.walls.length;i++){
        const at=i*4
        for(const [value,reference] of [[result[at]!,cpu.visited[i]!],[result[at+1]!,cpu.frontier[i]!],[result[at+2]!,cpu.distance[i]!]]){
          const delta=Math.abs(value-reference)
          if(!Number.isFinite(delta)||delta>0.00001)mismatches++
          maxError=Math.max(maxError,delta)
        }
      }
      checked=maze.walls.length*3;differences=mismatches
      fastValidated=true
      put('wires',checked.toLocaleString()+' state values, final readback')
      put('errors',String(mismatches))
      const path=cpu.visited[maze.goal]===1?reconstructPath(maze,cpu):[]
      put('path',path.length?String(path.length):'NOT FOUND')
      if(path.length&&mismatches===0){resident.setPath(path);resident.draw()}
      status.textContent=mismatches===0?'GPU-resident BFS verified against independent CPU BFS.':'GPU-vs-CPU BFS mismatch: '+mismatches
      document.body.dataset.status=mismatches===0?'pass':'fail'
      if(smoke){
        const ok=mismatches===0
        document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({
          ok,mode:'gpu',fast:true,frames:frame,count:maze.walls.length,checked,
          mismatches,maxError,cpuMs,gpuMs})
        paused=true
      }
    }
    return
  }
  if(cpu.finished&&gpu.finished&&!smoke)return
  const startedCpu=performance.now()
  cpu=referenceWave(maze,cpu)
  cpuMs=performance.now()-startedCpu
  const input=mazeInputs(maze,gpu)
  const startedGpu=performance.now()
  const values=await evaluator!.evaluate(input)
  gpuMs=performance.now()-startedGpu
  if(smoke||frame%6===0)compare(input,values)
  gpu=circuitWave(maze,gpu,values,width,outputWires)
  frame++
  draw(cpuCanvas,cpuContext,cpu);draw(gpuCanvas,gpuContext,gpu)
  stepRequested=false
  put('depth',String(gpu.depth))
  put('explored',gpu.visited.reduce((sum,value)=>sum+value,0).toLocaleString())
  put('wires',checked.toLocaleString());put('errors',String(differences))
  put('path',gpu.visited[maze.goal]===1?String(reconstructPath(maze,gpu).length):'SEARCHING')
  put('cpu-info',cpuMs.toFixed(2)+' ms · '+cpu.depth+' expansions')
  put('gpu-info',gpuMs.toFixed(2)+' ms · '+gpu.depth+' expansions · '+evaluator!.mode.toUpperCase())
  if(cpu.finished&&gpu.finished)status.textContent='Search complete. Yellow marks the reconstructed shortest path.'
  document.body.dataset.status=differences===0?'pass':'fail'
  if(smoke&&frame>=8){
    const result={ok:differences===0&&evaluator!.mode==='gpu',mode:evaluator!.mode,frames:frame,
      count:maze.walls.length,operations:circuit.operations.length,checked,
      mismatches:differences,maxError,cpuMs,gpuMs}
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(result)
    document.body.dataset.status=result.ok?'pass':'fail'
    paused=true
  }
}
async function loop(){
  try{await tick()}catch(error){
    status.textContent='Experiment error: '+String(error)
    document.body.dataset.status='fail'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
    paused=true
  }
  if(!(smoke&&paused))requestAnimationFrame(()=>{void loop()})
}
document.querySelector<HTMLButtonElement>('#pause')!.onclick=()=>{
  paused=!paused;put('pause',paused?'RESUME':'PAUSE')}
document.querySelector<HTMLButtonElement>('#step')!.onclick=()=>{paused=true;stepRequested=true;put('pause','RESUME')}
document.querySelector<HTMLButtonElement>('#restart')!.onclick=()=>{
  restartRequested=true;paused=false;put('pause','PAUSE')}
size.onchange=()=>{restartRequested=true}
strategy.onchange=()=>{restartRequested=true}
engine.onchange=()=>{restartRequested=true}
void loop()

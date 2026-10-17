import {createWebGpuEvaluator,type WebGpuCircuitEvaluator} from '../../src/runtime/webgpu.ts'
import {buildMazeCircuit, MAZE_SOURCE_SIZE} from './model.ts'
import {beginSearch,circuitWave,cpuCircuitWave,generateMaze,mazeInputs,
  reconstructPath,referenceWave,type Maze,type SearchState} from './engine.ts'

const params=new URLSearchParams(location.search),smoke=params.get('smoke')==='1'
const {circuit,outputWires}=buildMazeCircuit()
const width=circuit.sourceSize+circuit.operations.length
const size=document.querySelector<HTMLSelectElement>('#size')!
const strategy=document.querySelector<HTMLSelectElement>('#strategy')!
const cpuCanvas=document.querySelector<HTMLCanvasElement>('#cpu')!
const gpuCanvas=document.querySelector<HTMLCanvasElement>('#gpu')!
const cpuContext=cpuCanvas.getContext('2d')!,gpuContext=gpuCanvas.getContext('2d')!
const status=document.querySelector<HTMLElement>('#status')!
const put=(id:string,value:string)=>{document.getElementById(id)!.textContent=value}
let seed=1849,maze:Maze=generateMaze(),cpu:SearchState=beginSearch(maze),gpu:SearchState=beginSearch(maze)
let evaluator:WebGpuCircuitEvaluator|null=null
let frame=0,paused=false,restartRequested=true,stepRequested=false
let differences=0,maxError=0,checked=0,watch=0
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
  const path=state.visited[maze.goal]?new Set(reconstructPath(maze,state)):new Set<number>()
  ctx.fillStyle='#080c1d';ctx.fillRect(0,0,bounds.width,bounds.height)
  for(let i=0;i<maze.walls.length;i++){
    if(maze.walls[i])continue
    const x=(i%maze.columns)*cellW,y=Math.floor(i/maze.columns)*cellH
    let color=state.frontier[i]?hues.frontier:state.visited[i]?hues.seen:hues.unseen
    if(path.has(i))color=hues.path
    if(i===maze.goal)color=hues.goal
    if(i===maze.start)color=hues.start
    ctx.fillStyle=color
    ctx.fillRect(x+gap,y+gap,cellW-2*gap,cellH-2*gap)
  }
}
async function restart(){
  evaluator?.destroy();evaluator=null
  const cols=Number(size.value),rows=cols===31?21:cols===61?39:31
  maze=generateMaze(cols,rows,seed++)
  cpu=beginSearch(maze);gpu=beginSearch(maze)
  frame=0;differences=0;maxError=0;checked=0;watch=0
  status.textContent='Compiling neural wavefront expansion…'
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
  if(restartRequested||evaluator===null)await restart()
  if(paused&&!stepRequested)return
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
  put('path',gpu.visited[maze.goal]?String(reconstructPath(maze,gpu).length):'SEARCHING')
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
void loop()

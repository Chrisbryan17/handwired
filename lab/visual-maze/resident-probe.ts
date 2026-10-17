import {ResidentMazeGpu} from './gpu-resident.ts'
import {generateMaze,beginSearch,referenceWave,reconstructPath} from './engine.ts'
const params=new URLSearchParams(location.search)
async function run(){
 const columns=Number(params.get('columns')??'47'),rows=columns===61?39:columns===31?21:31
 const maze=generateMaze(columns,rows,1849),canvas=document.querySelector<HTMLCanvasElement>('#gpu')!
 const started=performance.now()
 const backend=await ResidentMazeGpu.create(canvas,maze)
 const setupMs=performance.now()-started
 try{
   let cpu=beginSearch(maze),frames=0
   const t=performance.now()
   while(!cpu.finished&&frames<3000){
     cpu=referenceWave(maze,cpu)
     backend.frame()
     frames++
     if(frames%100===0)await backend.read()
   }
   const states=await backend.read()
   const elapsed=performance.now()-t
   let mismatches=0,maxError=0,nonfinite=0
   for(let i=0;i<maze.walls.length;i++){
     for(const [v,ref] of [[states[i*4]!,cpu.visited[i]!],
       [states[i*4+1]!,cpu.frontier[i]!],[states[i*4+2]!,cpu.distance[i]!]]){
       const delta=Math.abs(v-ref)
       if(!Number.isFinite(delta))nonfinite++
       if(!Number.isFinite(delta)||delta>1e-5)mismatches++
       maxError=Math.max(maxError,delta)
     }
   }
   const path=cpu.visited[maze.goal]===1?reconstructPath(maze,cpu):[]
   if(path.length>0&&mismatches===0){backend.setPath(path);backend.draw()}
   const ok=cpu.finished&&cpu.visited[maze.goal]===1&&frames===backend.steps&&mismatches===0
   const report={ok,mode:'gpu',frames,count:maze.walls.length,
     checked:maze.walls.length*3,mismatches,maxError,nonfinite,pathLength:path.length,
     setupMs,wallMs:elapsed,msPerFrame:elapsed/Math.max(1,frames),columns,rows,
     algorithm:'exact Handwired BFS with fully GPU-resident neighbor gathering'}
   document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
   document.querySelector<HTMLElement>('#status')!.textContent=ok?'Full GPU BFS and shortest path validated':'BFS mismatch'
   document.body.dataset.status=ok?'pass':'fail'
 } finally {backend.destroy()}
}
void run().catch(e=>{
 document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(e)})
 document.querySelector<HTMLElement>('#status')!.textContent=String(e)
 document.body.dataset.status='fail'
})

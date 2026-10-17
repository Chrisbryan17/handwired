import {ResidentSwarmGpu} from './gpu-resident.ts'
import {initialPopulation} from './simulation.ts'
const query=new URLSearchParams(location.search)
const count=Number(query.get('count')??'1024')
const canvas=document.querySelector<HTMLCanvasElement>('#gpu')!
async function main(){
  const started=performance.now()
  const swarm=await ResidentSwarmGpu.create(canvas,count)
  try{
    const setupMs=performance.now()-started
    const before=initialPopulation(count).state
    const frames=Number(query.get('frames')??'8')
    if(!Number.isInteger(frames)||frames<1||frames>500)throw new Error('invalid smoke frame count')
    const startedFrames=performance.now()
    for(let i=0;i<frames;i++){
      swarm.frame({mode:'off',x:.5,y:.5},canvas.width,canvas.height)
      // Ensure the measurement includes actual GPU work, not queue submission alone.
      if(i%8===7)await swarm.device.queue.onSubmittedWorkDone()
    }
    await swarm.device.queue.onSubmittedWorkDone()
    const frameWallMs=performance.now()-startedFrames
    const state=await swarm.readState()
    let nonfinite=0,outOfBounds=0,moved=0,maxDisplacement=0
    for(let i=0;i<count;i++){
      const p=i*4,x=state[p]!,y=state[p+1]!,vx=state[p+2]!,vy=state[p+3]!
      if(![x,y,vx,vy].every(Number.isFinite))nonfinite++
      if(x<0||x>1||y<0||y>1)outOfBounds++
      const distance=Math.hypot(x-before[p]!,y-before[p+1]!)
      if(distance>0.00001)moved++
      maxDisplacement=Math.max(maxDisplacement,distance)
    }
    const ok=nonfinite===0&&outOfBounds===0&&moved===count
    const report={ok,mode:'gpu',count,frames,checked:count*4,nonfinite,outOfBounds,
      moved,maxDisplacement,setupMs,frameWallMs,msPerFrame:frameWallMs/frames,
      device:'real WebGPU required',algorithm:'cell-aggregated approximate Boids with Handwired circuit'}
    document.body.dataset.status=ok?'pass':'fail'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
    document.querySelector<HTMLElement>('#status')!.textContent=ok?'Actual GPU resident swarm PASS':'GPU swarm FAIL'
  }finally{swarm.destroy()}
}
void main().catch(error=>{
  document.body.dataset.status='fail'
  document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
  document.querySelector<HTMLElement>('#status')!.textContent=String(error)
})

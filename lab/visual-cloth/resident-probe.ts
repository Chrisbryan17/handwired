import {makeCloth,cpuClothFrame} from './physics.ts'
import {ResidentClothGpu} from './gpu-resident.ts'
async function main(){
  const start=makeCloth(15,11)
  let cpu=start
  const gpu=await ResidentClothGpu.create(start)
  try{
    gpu.load(start)
    const began=performance.now()
    for(let i=0;i<8;i++){
      cpu=cpuClothFrame(cpu,{dragged:null,x:.5,y:.5,wind:1.2}).next
      await gpu.step({wind:1.2})
    }
    const gpuMs=performance.now()-began
    const values=await gpu.read()
    let pinnedErrors=0,nonfinite=0,maxError=0,sumSquared=0
    for(let i=0;i<start.columns*start.rows;i++){
      const offset=i*4
      const x=values[offset]!,y=values[offset+1]!
      if(!Number.isFinite(x)||!Number.isFinite(y))nonfinite++
      if(start.pinned[i]===1&&
        (Math.abs(x-start.origin[i*2]!)>1e-6||
          Math.abs(y-start.origin[i*2+1]!)>1e-6))pinnedErrors++
      for(let j=0;j<2;j++){
        const delta=Math.abs(values[offset+j]!-cpu.state[offset+j]!)
        maxError=Math.max(maxError,delta);sumSquared+=delta*delta
      }
    }
    const rms=Math.sqrt(sumSquared/(start.columns*start.rows*2))
    const midpoint=start.springs[0]!
    const mx=(values[midpoint.a*4]!+values[midpoint.b*4]!)/2
    const my=(values[midpoint.a*4+1]!+values[midpoint.b*4+1]!)/2
    const initialTears=await gpu.readTearMasks()
    await gpu.step({wind:1.2,cut:true,x:mx,y:my,tearRadius:.045})
    const newTears=await gpu.readTearMasks()
    const originalBroken=initialTears.reduce((n,mask)=>n+(mask!==0?1:0),0)
    const broken=newTears.reduce((n,mask)=>n+(mask!==0?1:0),0)
    const middle=Math.floor(start.rows/2)*start.columns+Math.floor(start.columns/2)
    await gpu.step({wind:0,dragged:middle,x:.63,y:.22})
    const draggedPosition=await gpu.read()
    const dragError=Math.hypot(draggedPosition[middle*4]!-.63,draggedPosition[middle*4+1]!-.22)
    const report={ok:nonfinite===0&&pinnedErrors===0&&maxError<.14&&broken>originalBroken&&dragError<1e-6,
      mode:'gpu',frames:8,count:start.columns*start.rows,
      checked:start.columns*start.rows*2,mismatches:nonfinite+pinnedErrors,
      maxError,rms,pinnedErrors,nonfinite,gpuMs,numericTolerance:.14,
      broken,originalBroken,dragError,interactionSteps:2}
    document.querySelector<HTMLElement>('#status')!.textContent=report.ok?
      'GPU-resident cloth execution succeeded.':'GPU-resident cloth comparison failed.'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
    document.body.dataset.status=report.ok?'pass':'fail'
  } finally {gpu.destroy()}
}
void main().catch(error=>{
  document.body.dataset.status='fail'
  document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
  document.querySelector<HTMLElement>('#status')!.textContent=String(error)
})

import {createFluid,cpuFluidFrame} from './physics.ts'
import {ResidentFluidGpu} from './gpu-resident.ts'

async function main(){
  const {width,height}={width:24,height:16}
  let cpu=createFluid(width,height)
  const gpu=await ResidentFluidGpu.create(width,height)
  const init=cpu.fields.slice()
  const started=performance.now()
  try{
    gpu.load(init)
    for(let n=0;n<8;n++){
      cpu=cpuFluidFrame(cpu).next
      await gpu.step()
    }
    const gpuMs=performance.now()-started
    const gpuResult=await gpu.read()
    let maxError=0,sumSquared=0,nonfinite=0
    for(let i=0;i<gpuResult.length;i++){
      const actual=gpuResult[i]!,reference=cpu.fields[i]!
      if(!Number.isFinite(actual))nonfinite++
      const difference=Math.abs(actual-reference)
      maxError=Math.max(maxError,difference)
      sumSquared+=difference*difference
    }
    const rms=Math.sqrt(sumSquared/gpuResult.length)
    const differences=gpuResult.reduce((total,value,index)=>total+(
      Math.abs(value-cpu.fields[index]!)>0.0001?1:0),0)
    let dyeBefore=0
    for(let i=0;i<width*height;i++)dyeBefore+=gpuResult[i*5+2]!+gpuResult[i*5+3]!+gpuResult[i*5+4]!
    await gpu.step({x:.49,y:.48,dx:.08,dy:-.06,strength:2})
    const injected=await gpu.read()
    let dyeAfter=0
    for(let i=0;i<width*height;i++)dyeAfter+=injected[i*5+2]!+injected[i*5+3]!+injected[i*5+4]!
    for(let i=0;i<24;i++)await gpu.step()
    const longRun=await gpu.read()
    const longRunFinite=longRun.every(Number.isFinite)
    const report={ok:nonfinite===0&&differences===0&&longRunFinite&&dyeAfter>dyeBefore,
      mode:'gpu',frames:8,
      count:width*height,checked:gpuResult.length,mismatches:differences,
      maxError,rms,nonfinite,gpuMs,adapterInfo:gpu.adapterInfo,
      numericTolerance:.0001,dyeBefore,dyeAfter,longRunFinite,additionalSteps:25}
    document.querySelector<HTMLElement>('#status')!.textContent=
      report.ok?'Real WebGPU fused neural fluid completed.':'Hardware numerical comparison failed.'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
    document.body.dataset.status=report.ok?'pass':'fail'
  }finally{gpu.destroy()}
}
void main().catch(error=>{
  document.body.dataset.status='fail'
  document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
  document.querySelector<HTMLElement>('#status')!.textContent='WebGPU failure: '+String(error)
})

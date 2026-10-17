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
      Math.abs(value-cpu.fields[index]!)>0.12?1:0),0)
    const report={ok:nonfinite===0&&differences===0,mode:'gpu',frames:8,
      count:width*height,checked:gpuResult.length,mismatches:differences,
      maxError,rms,nonfinite,gpuMs,adapterInfo:gpu.adapterInfo,
      numericTolerance:.12}
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

import {ResidentClothGpu} from './gpu-resident.ts'
import {makeCloth} from './physics.ts'
const options=new URLSearchParams(location.search)
async function run(){
  const columns=Number(options.get('size')??'33'),rows=columns===49?35:columns===25?17:23
  const frames=Number(options.get('frames')??'120')
  if(!Number.isInteger(frames)||frames<8||frames>240)throw new Error('invalid benchmark')
  const canvas=document.querySelector<HTMLCanvasElement>('#gpu')!
  const cloth=makeCloth(columns,rows),start=performance.now()
  const engine=await ResidentClothGpu.create(cloth)
  const setupMs=performance.now()-start
  try{
    const initial=cloth
    engine.load(initial)
    await engine.attachCanvas(canvas)
    const t=performance.now()
    for(let i=0;i<frames;i++)engine.frame({wind:1.2})
    const values=await engine.read()
    const wallMs=performance.now()-t
    let nonfinite=0,outOfBounds=0,pinnedErrors=0
    for(let i=0;i<columns*rows;i++){
      const x=values[i*4]!,y=values[i*4+1]!
      if(!Number.isFinite(x)||!Number.isFinite(y))nonfinite++
      if(x<0||x>1||y<0||y>1)outOfBounds++
      if(cloth.pinned[i]===1&&
        (Math.abs(x-cloth.origin[i*2]!)>1e-6||Math.abs(y-cloth.origin[i*2+1]!)>1e-6))pinnedErrors++
    }
    engine.load(initial)
    const baselineStarted=performance.now()
    let baseline:Float32Array|undefined
    for(let i=0;i<frames;i++){
      await engine.step({wind:1.2})
      baseline=await engine.read()
    }
    const baselineWallMs=performance.now()-baselineStarted
    let maxParityError=0
    for(let i=0;i<values.length;i++){
      maxParityError=Math.max(maxParityError,Math.abs(values[i]!-baseline![i]!))
    }
    const baselineMsPerFrame=baselineWallMs/frames
    const speedup=baselineWallMs/wallMs
    const ok=nonfinite===0&&outOfBounds===0&&pinnedErrors===0&&maxParityError<1e-5
    const report={ok,mode:'gpu',frames,count:columns*rows,checked:values.length,
      mismatches:nonfinite+outOfBounds+pinnedErrors,nonfinite,outOfBounds,pinnedErrors,
      setupMs,wallMs,msPerFrame:wallMs/frames,baselineWallMs,baselineMsPerFrame,speedup,maxParityError,
      measurement:'GPU-resident cloth solver + direct mesh render + ONE final readback'}
    document.body.dataset.status=ok?'pass':'fail'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
    document.querySelector<HTMLElement>('#status')!.textContent=ok?'PASS':'FAIL'
  }finally{engine.destroy()}
}
void run().catch(error=>{
  document.body.dataset.status='fail'
  document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
})

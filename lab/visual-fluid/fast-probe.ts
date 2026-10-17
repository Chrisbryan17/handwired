import {ResidentFluidGpu} from './gpu-resident.ts'
import {createFluid} from './physics.ts'
const options=new URLSearchParams(location.search)
async function run(){
  const width=Number(options.get('size')??'64'),height=width===96?64:width===128?80:40
  const frames=Number(options.get('frames')??'120')
  if(!Number.isInteger(frames)||frames<8||frames>240||width*height>8192)throw new Error('invalid test dimensions')
  const canvas=document.querySelector<HTMLCanvasElement>('#gpu')!
  const start=performance.now()
  const engine=await ResidentFluidGpu.create(width,height)
  const setupMs=performance.now()-start
  try{
    engine.load(createFluid(width,height))
    await engine.attachCanvas(canvas)
    const t=performance.now()
    for(let i=0;i<frames;i++)engine.frame()
    // Only one GPU readback at the end; includes GPU completion.
    const field=await engine.read()
    const wallMs=performance.now()-t
    let nonfinite=0,outOfBounds=0,brightness=0
    for(let i=0;i<field.length;i++){
      const v=field[i]!
      if(!Number.isFinite(v))nonfinite++
      if(Math.abs(v)>6.001)outOfBounds++
      if(i%5>=2)brightness+=v
    }
    const ok=nonfinite===0&&outOfBounds===0&&brightness>0
    const report={ok,mode:'gpu',frames,count:width*height,checked:field.length,
      mismatches:nonfinite+outOfBounds,nonfinite,outOfBounds,
      brightness,setupMs,wallMs,msPerFrame:wallMs/frames,
      measurement:'GPU-resident fluid compute + direct render + ONE final readback'}
    document.body.dataset.status=ok?'pass':'fail'
    document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify(report)
    document.querySelector<HTMLElement>('#status')!.textContent=ok?'PASS':'FAIL'
  }finally{engine.destroy()}
}
void run().catch(error=>{
  document.body.dataset.status='fail'
  document.querySelector<HTMLElement>('#smoke')!.textContent=JSON.stringify({ok:false,error:String(error)})
})

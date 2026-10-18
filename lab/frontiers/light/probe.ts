import {LightGpuRenderer} from './solver.ts'
import {cameraRay,tracePathCpu} from './model.ts'
const node=document.getElementById('result')
const canvas=document.getElementById('scene') as HTMLCanvasElement|null
if(canvas===null)throw new Error('light probe has no canvas')
let renderer:LightGpuRenderer|null=null
try{
  renderer=await LightGpuRenderer.create(canvas,160,120)
  for(let i=0;i<12;i++)renderer.frame()
  await renderer.flush()
  const raw=await renderer.readAccumulation()
  let invalid=0,nonzero=0,max=0,sum=0
  for(let i=0;i<raw.length;i+=4){
    const r=raw[i]!,g=raw[i+1]!,b=raw[i+2]!,samples=raw[i+3]!
    if(!Number.isFinite(r)||!Number.isFinite(g)||!Number.isFinite(b)||samples!==12)invalid++
    const intensity=r+g+b
    if(intensity>0)nonzero++
    max=Math.max(max,intensity);sum+=intensity
  }
  const cpuSample=tracePathCpu(cameraRay(80,60,160,120),37)
  const summary={ok:invalid===0&&nonzero>100&&max>0,
    adapter:renderer.adapterName,samples:renderer.samplesPerPixel,
    pixels:160*120,invalid,nonzero,maxLinearRadianceSum:max,
    meanLinearRadianceSum:sum/(160*120*12),
    cpuReferenceSample:cpuSample,hardwareExpected:true}
  if(node)node.textContent=JSON.stringify(summary)
}catch(error){
  if(node)node.textContent=JSON.stringify({ok:false,error:String(error)})
}finally{renderer?.dispose()}

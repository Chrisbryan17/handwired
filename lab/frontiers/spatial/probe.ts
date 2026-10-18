import {seededSpatialPoints,countSpatialNeighborsCpu,
  type SpatialGridConfig} from '../../../src/runtime/frontiers-spatial.ts'
import {SpatialGridGpu} from '../../../src/runtime/frontiers-spatial-gpu.ts'
const report=document.getElementById('result')
type Scenario={name:string,points:Float32Array,config:SpatialGridConfig}
const scattered:SpatialGridConfig={origin:[0,0,0],dimensions:[10,10,10],cellSize:1,radius:.75}
const packed:SpatialGridConfig={origin:[0,0,0],dimensions:[4,4,4],cellSize:.5,radius:.4}
const clustered=seededSpatialPoints(128,11)
for(let i=0;i<128;i++){
  clustered[i*4]=Math.fround(.46+(i%7)*.044)
  clustered[i*4+1]=Math.fround(.43+(i%9)*.033)
  clustered[i*4+2]=Math.fround(.45+(i%5)*.036)
}
clustered[0]=-1 // Explicit out-of-domain point, excluded from all queries.
const scenarios:Scenario[]=[
 {name:'uniform-384',points:seededSpatialPoints(384,87),config:scattered},
 {name:'dense-127-plus-outside',points:clustered,config:packed},
]
const results=[]
let adapter=''
try{
 for(const scenario of scenarios){
  const reference=countSpatialNeighborsCpu(scenario.points,scenario.config)
  const gpu=await SpatialGridGpu.create(scenario.points,scenario.config)
  try{
   adapter=gpu.gpu.adapter.name
   const t0=performance.now()
   gpu.queryNow()
   await gpu.flush()
   const synchronizedMs=performance.now()-t0
   const t1=performance.now()
   const actual=await gpu.readCounts()
   const readbackMs=performance.now()-t1
   let mismatches=0,maxAbsoluteError=0,total=0
   for(let i=0;i<reference.length;i++){
    const error=Math.abs(reference[i]!-actual[i]!)
    if(error!==0)mismatches++
    maxAbsoluteError=Math.max(maxAbsoluteError,error)
    total+=actual[i]!
   }
   results.push({name:scenario.name,points:reference.length,
    cells:gpu.cellCount,mismatches,maxAbsoluteError,totalNeighborCount:total,
    synchronizedMs,readbackMs,trackedGpuBytes:gpu.gpu.bytesAllocated(),
    gpuBuildVersions:gpu.version})
  }finally{gpu.dispose()}
 }
 const ok=results.every(item=>item.mismatches===0)
 if(report)report.textContent=JSON.stringify({ok,adapter,algorithm:
  'WebGPU atomic linked-cell scatter and 27-cell Euclidean radius queries',
  oracle:'independent CPU O(N^2) exact neighbor counts',results,
  hardwareExpected:true,gpuTiming:'host command + queue completion, not timestamp-query'} )
}catch(error){
 if(report)report.textContent=JSON.stringify({ok:false,error:String(error)})
}

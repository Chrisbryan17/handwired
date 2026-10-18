import {cameraRay,intersectScene,type Ray} from './model.ts'
import {lightKernelWgsl, type LightGpuRenderer} from './solver.ts'

/** Compare actual WGSL scene traversal against an independently handwritten
 * CPU primitive-intersection oracle. This is deterministic ray geometry parity,
 * not a claim of pixel-exact Monte Carlo radiance agreement.
 */
const probeWgsl = `
struct ProbeInput {origin:vec4<f32>, direction:vec4<f32>};
@group(1) @binding(0) var<storage,read> rayInputs:array<ProbeInput>;
@group(1) @binding(1) var<storage,read_write> rayResults:array<vec4<f32>>;
@compute @workgroup_size(64)
fn geometryProbe(@builtin(global_invocation_id) tid:vec3<u32>) {
  if(tid.x >= arrayLength(&rayResults)){return;}
  let p=rayInputs[tid.x];
  let h=traceScene(Ray(p.origin.xyz,normalize(p.direction.xyz)),1.0e20);
  let distance=select(h.t,-1.0,h.material==9u);
  rayResults[tid.x]=vec4<f32>(distance,f32(h.material),h.normal.x,h.normal.y);
}
`
const kind = (name:string|undefined):number =>
  name==='diffuse'?0:name==='mirror'?1:name==='dielectric'?2:name==='emissive'?3:9

export async function verifyLightGpuGeometry(renderer:LightGpuRenderer){
  const gpu=renderer.gpu
  const device=gpu.device
  const rayList:Ray[]=[]
  // A reproducible camera-plane grid covering intersections and misses.
  for(let y=3;y<120;y+=11)for(let x=4;x<160;x+=13)
    rayList.push(cameraRay(x,y,160,120))
  // Additional controlled cardinal and near-normal rays.
  rayList.push({origin:[0,0,-2.85],direction:[0,1,0]})
  rayList.push({origin:[0,0,-3],direction:[1,0,0]})
  rayList.push({origin:[0,0,-3],direction:[0,-1,0]})
  rayList.push({origin:[0,0,2.4],direction:[0,0,1]}) // guaranteed camera-side miss
  const count=rayList.length
  const source=new Float32Array(count*8)
  for(let i=0;i<count;i++){
    const ray=rayList[i]!,at=i*8
    source.set([...ray.origin,0,...ray.direction,0],at)
  }
  const inBuffer=gpu.allocate(source.byteLength,128|8,'Oracle ray origins/directions')
  const outBuffer=gpu.allocate(count*16,128|4,'GPU intersection results')
  const uniform=gpu.allocate(48,64|8,'Oracle scene parameters')
  device.queue.writeBuffer(inBuffer,0,source)
  const values=new Float32Array(12)
  values[7]=1 // Physically non-zero emitter intensity.
  device.queue.writeBuffer(uniform,0,values)
  const shader=device.createShaderModule({
    code:lightKernelWgsl+'\n'+probeWgsl,label:'Independent light scene intersection probe',
  })
  const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
  if(errors.length)throw new Error('Intersection probe WGSL: '+errors.map(x=>x.message).join('; '))
  const pipeline=await device.createComputePipelineAsync({
    layout:'auto',compute:{module:shader,entryPoint:'geometryProbe'},
  })
  const bind0=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:1,resource:{buffer:uniform}},
  ]})
  const bind1=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[
    {binding:0,resource:{buffer:inBuffer}},
    {binding:1,resource:{buffer:outBuffer}},
  ]})
  const stage=device.createBuffer({size:count*16,usage:8|1})
  try{
    const encoder=device.createCommandEncoder()
    const pass=encoder.beginComputePass()
    pass.setPipeline(pipeline);pass.setBindGroup(0,bind0);pass.setBindGroup(1,bind1)
    pass.dispatchWorkgroups(Math.ceil(count/64));pass.end()
    encoder.copyBufferToBuffer(outBuffer,0,stage,0,count*16)
    device.queue.submit([encoder.finish()])
    await stage.mapAsync(1)
    const actual=new Float32Array(stage.getMappedRange().slice(0))
    stage.unmap()
    let mismatches=0,maxTError=0,maxNormalError=0,missedRays=0,hits=0
    let lightHits=0,mirrorHits=0,glassHits=0
    for(let i=0;i<count;i++){
      const expected=intersectScene(rayList[i]!)
      const at=i*4
      const t=actual[at]!,material=actual[at+1]!
      if(!Number.isFinite(t)||material!==kind(expected?.material)){mismatches++;continue}
      if(expected===null){missedRays++;if(t!==-1)mismatches++;continue}
      hits++
      if(expected.material==='emissive')lightHits++
      if(expected.material==='mirror')mirrorHits++
      if(expected.material==='dielectric')glassHits++
      const tError=Math.abs(t-expected.t)
      const normalError=Math.max(Math.abs(actual[at+2]!-expected.normal[0]),
        Math.abs(actual[at+3]!-expected.normal[1]))
      maxTError=Math.max(maxTError,tError)
      maxNormalError=Math.max(maxNormalError,normalError)
      if(tError>0.002||normalError>0.002)mismatches++
    }
    return {ok:mismatches===0&&hits>0&&missedRays>0&&mirrorHits>0&&glassHits>0&&lightHits>0,
      rays:count,hits,missedRays,lightHits,mirrorHits,glassHits,
      mismatches,maxTError,maxNormalError,
      positionTolerance:0.002,normalTolerance:0.002,
      note:'GPU scene intersection compared to independent CPU ray oracle, not full stochastic path/radiance parity'}
  }finally{stage.destroy()}
}

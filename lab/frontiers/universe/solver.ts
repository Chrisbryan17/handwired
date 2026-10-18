import {emitInlineCircuitWgsl} from '../../../src/runtime/webgpu-inline.ts'
import {requestStrictGpu,type GpuAllocation} from '../../../src/runtime/frontiers-device.ts'
import {buildUniverseCircuit,checkUniverseState,STRIDE,
  type UniverseParams,type UniverseState} from './model.ts'

/**
 * Actual GPU-resident O(N²) softened Newtonian force summation.
 * Each body reads all other masses, accumulates gravitational acceleration,
 * then calls a function emitted from a Handwired Circuit for kick + drift.
 * Physics and integration remain GPU-resident through repeated encodeStep().
 * Scientific limitation: fixed-step symplectic Euler, FP32, nonadaptive and
 * no Barnes-Hut/FMM. No claim of millions of *simulated* bodies.
 */
const integration = buildUniverseCircuit()
export const universeKernelWgsl = `
struct Particle {position:vec4<f32>, velocity:vec4<f32>};
struct Params {count:u32, gravity:f32, softening:f32, pad:f32};
@group(0) @binding(0) var<storage,read> bodies:array<Particle>;
@group(0) @binding(1) var<storage,read_write> nextBodies:array<Particle>;
@group(0) @binding(2) var<uniform> parameters:Params;
` + emitInlineCircuitWgsl(integration.circuit,integration.outputWires,'handwiredKickDrift') + `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) thread:vec3<u32>) {
  let i=thread.x;
  if (i>=parameters.count){return;}
  let particle=bodies[i];
  var acceleration=vec3<f32>(0.0);
  let epsilon2=parameters.softening*parameters.softening;
  for(var j=0u;j<parameters.count;j=j+1u) {
    if(i==j){continue;}
    let other=bodies[j];
    let offset=other.position.xyz-particle.position.xyz;
    let distance2=dot(offset,offset)+epsilon2;
    let weight=parameters.gravity*other.position.w*inverseSqrt(distance2)/distance2;
    acceleration=acceleration+offset*weight;
  }
  var source:array<f32,9>;
  source[0u]=particle.position.x; source[1u]=particle.position.y;
  source[2u]=particle.position.z;
  source[3u]=particle.velocity.x; source[4u]=particle.velocity.y;
  source[5u]=particle.velocity.z;
  source[6u]=acceleration.x; source[7u]=acceleration.y;
  source[8u]=acceleration.z;
  let result=handwiredKickDrift(source);
  nextBodies[i].position=vec4<f32>(result[0u],result[1u],result[2u],particle.position.w);
  nextBodies[i].velocity=vec4<f32>(result[3u],result[4u],result[5u],particle.velocity.w);
}
`

export class UniverseGpuSolver {
  readonly gpu:GpuAllocation
  readonly count:number
  readonly params:UniverseParams
  readonly buffers:readonly [GPUBuffer,GPUBuffer]
  readonly computeSource=universeKernelWgsl
  private current=0
  private tick=0
  private disposed=false
  private readonly uniforms:GPUBuffer
  private readonly pipeline:GPUComputePipeline
  private readonly groups:readonly [GPUBindGroup,GPUBindGroup]

  private constructor(gpu:GpuAllocation,state:UniverseState,params:UniverseParams,
    buffers:[GPUBuffer,GPUBuffer],uniforms:GPUBuffer,
    pipeline:GPUComputePipeline,groups:[GPUBindGroup,GPUBindGroup]) {
    this.gpu=gpu;this.count=state.length/STRIDE;this.params={...params}
    this.buffers=buffers;this.uniforms=uniforms;this.pipeline=pipeline;this.groups=groups
  }
  static async create(state:UniverseState,params:UniverseParams):Promise<UniverseGpuSolver> {
    const count=checkUniverseState(state,params)
    if(count>4096)throw new Error('Direct N-body upper safety limit is 4096 bodies')
    const gpu=await requestStrictGpu('Handwired Neural Universe')
    try {
      if(Math.ceil(count/64)>gpu.device.limits.maxComputeWorkgroupsPerDimension)
        throw new Error('N-body exceeds dispatch dimension limit')
      const bits={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64}
      const bytes=state.byteLength
      const buffers:[GPUBuffer,GPUBuffer]=[
        gpu.allocate(bytes,bits.STORAGE|bits.COPY_SRC|bits.COPY_DST,'N-body A'),
        gpu.allocate(bytes,bits.STORAGE|bits.COPY_SRC|bits.COPY_DST,'N-body B'),
      ]
      gpu.device.queue.writeBuffer(buffers[0],0,state)
      const uniforms=gpu.allocate(16,bits.UNIFORM|bits.COPY_DST,'N-body parameters')
      const module=gpu.device.createShaderModule({code:universeKernelWgsl,label:'Handwired gravity and kick-drift'})
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error')
      if(errors.length)throw new Error('N-body WGSL: '+errors.map(m=>m.message).join('; '))
      const pipeline=await gpu.device.createComputePipelineAsync({
        layout:'auto',compute:{module,entryPoint:'main'},
      })
      const group=(source:GPUBuffer,target:GPUBuffer)=>gpu.device.createBindGroup({
        layout:pipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:source}},
          {binding:1,resource:{buffer:target}},
          {binding:2,resource:{buffer:uniforms}},
        ],
      })
      const result=new UniverseGpuSolver(gpu,state,params,buffers,uniforms,pipeline,
        [group(buffers[0],buffers[1]),group(buffers[1],buffers[0])])
      result.setParameters(params)
      return result
    }catch(error){gpu.dispose();throw error}
  }
  private check(){
    if(this.disposed)throw new Error('N-body GPU solver disposed')
    this.gpu.check()
  }
  get frame(){return this.tick}
  get currentBuffer(){return this.buffers[this.current]!}
  get currentIndex(){return this.current}
  setParameters(params:UniverseParams) {
    this.check()
    if(!Number.isFinite(params.gravity)||params.gravity<0||params.gravity>5||
      !Number.isFinite(params.softening)||params.softening<.002||params.softening>.5)
      throw new Error('invalid N-body solver parameters')
    this.params.gravity=params.gravity;this.params.softening=params.softening
    const data=new ArrayBuffer(16),view=new DataView(data)
    view.setUint32(0,this.count,true)
    view.setFloat32(4,params.gravity,true)
    view.setFloat32(8,params.softening,true)
    this.gpu.device.queue.writeBuffer(this.uniforms,0,data)
  }
  encodeStep(encoder:GPUCommandEncoder) {
    this.check()
    const pass=encoder.beginComputePass({label:'All-pairs gravity + Handwired integration'})
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0,this.groups[this.current]!)
    pass.dispatchWorkgroups(Math.ceil(this.count/64))
    pass.end()
    this.current=1-this.current
    this.tick++
  }
  step(steps=1) {
    this.check()
    if(!Number.isSafeInteger(steps)||steps<1||steps>64)throw new Error('invalid step count')
    const encoder=this.gpu.device.createCommandEncoder()
    for(let i=0;i<steps;i++)this.encodeStep(encoder)
    this.gpu.device.queue.submit([encoder.finish()])
  }
  reset(state:UniverseState) {
    this.check()
    if(state.length!==this.count*STRIDE)throw new Error('reset count mismatch')
    checkUniverseState(state,this.params)
    this.gpu.device.queue.writeBuffer(this.buffers[0],0,state)
    this.current=0;this.tick=0
  }
  /** Explicit synchronized diagnostic readback. Never called in the render loop. */
  async read():Promise<UniverseState> {
    this.check()
    const bytes=this.count*STRIDE*4
    const stage=this.gpu.device.createBuffer({size:bytes,
      usage:8|1})
    try{
      const encoder=this.gpu.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.currentBuffer,0,stage,0,bytes)
      this.gpu.device.queue.submit([encoder.finish()])
      await stage.mapAsync(1)
      const data=new Float32Array(stage.getMappedRange().slice(0))
      stage.unmap()
      this.check()
      return data
    }finally{stage.destroy()}
  }
  async flush(){
    this.check()
    await this.gpu.device.queue.onSubmittedWorkDone()
    this.check()
  }
  dispose(){if(this.disposed)return;this.disposed=true;this.gpu.dispose()}
}

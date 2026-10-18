import {emitInlineCircuitWgsl} from '../../../src/runtime/webgpu-inline.ts'
import {requestStrictGpu,type GpuAllocation} from '../../../src/runtime/frontiers-device.ts'
import {buildGenesisCircuit,genesisDefaults,validateGenesis,
  type GenesisField,type GenesisParams} from './model.ts'

/** GPU-resident 9-point periodic Gray–Scott reaction diffusion.
 * The generated Handwired function performs the final two-channel update.
 * The local Laplacian and nonlinear UV² reaction are domain-specific WGSL.
 * This is deterministic morphogenesis, not evolutionary training.
 */
const compiled=buildGenesisCircuit()
export const genesisKernelWgsl=`
struct Params{
  width:u32,height:u32,feed:f32,kill:f32,
  diffusionU:f32,diffusionV:f32,brushX:f32,brushY:f32,
  brushMode:u32,brushRadius:f32,pad0:f32,pad1:f32,
};
@group(0) @binding(0) var<storage,read> current:array<f32>;
@group(0) @binding(1) var<storage,read_write> next:array<f32>;
@group(0) @binding(2) var<uniform> params:Params;
fn concentration(x:i32,y:i32,ch:u32)->f32{
  let cx=u32((x%i32(params.width)+i32(params.width))%i32(params.width));
  let cy=u32((y%i32(params.height)+i32(params.height))%i32(params.height));
  return current[(cy*params.width+cx)*2u+ch];
}
fn laplacian(x:i32,y:i32,ch:u32)->f32 {
  let middle=concentration(x,y,ch);
  let axial=(concentration(x-1,y,ch)+concentration(x+1,y,ch)+
    concentration(x,y-1,ch)+concentration(x,y+1,ch))*.2;
  let diagonal=(concentration(x-1,y-1,ch)+concentration(x+1,y-1,ch)+
    concentration(x-1,y+1,ch)+concentration(x+1,y+1,ch))*.05;
  return -middle+axial+diagonal;
}
`+emitInlineCircuitWgsl(compiled.circuit,compiled.outputWires,'handwiredMorphogen')+`
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
  if(tid.x>=params.width||tid.y>=params.height){return;}
  let i=(tid.y*params.width+tid.x)*2u;
  var u=current[i];var v=current[i+1u];
  let lapU=laplacian(i32(tid.x),i32(tid.y),0u);
  let lapV=laplacian(i32(tid.x),i32(tid.y),1u);
  let uvv=u*v*v;
  var source:array<f32,7>;
  source[0u]=u;source[1u]=v;
  source[2u]=lapU*params.diffusionU/0.16;
  source[3u]=lapV*params.diffusionV/0.08;
  source[4u]=uvv;
  source[5u]=params.feed*(1.0-u);
  source[6u]=(params.feed+params.kill)*v;
  let changed=handwiredMorphogen(source);
  var outU=clamp(changed[0u],0.0,1.0);
  var outV=clamp(changed[1u],0.0,1.0);
  if(params.brushMode!=0u){
    let dx=f32(tid.x)-params.brushX*f32(params.width);
    let dy=f32(tid.y)-params.brushY*f32(params.height);
    if(dx*dx+dy*dy<params.brushRadius*params.brushRadius){
      outU=select(.50,1.0,params.brushMode==2u);
      outV=select(.27,0.0,params.brushMode==2u);
    }
  }
  next[i]=outU;next[i+1u]=outV;
}`
const usage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128}
export type GenesisBrush={x:number,y:number,mode:0|1|2,radius:number}

export class GenesisGpuSolver{
  readonly gpu:GpuAllocation
  readonly buffers:readonly [GPUBuffer,GPUBuffer]
  readonly width:number
  readonly height:number
  readonly params:GenesisParams
  private readonly parameterBuffer:GPUBuffer
  private readonly pipeline:GPUComputePipeline
  private readonly groups:[GPUBindGroup,GPUBindGroup]
  private index=0
  private iteration=0
  private disposed=false
  private brush:GenesisBrush={x:.5,y:.5,mode:0,radius:8}
  private constructor(gpu:GpuAllocation,field:GenesisField,params:GenesisParams,
    buffers:[GPUBuffer,GPUBuffer],parameters:GPUBuffer,pipeline:GPUComputePipeline,
    groups:[GPUBindGroup,GPUBindGroup]){
    this.gpu=gpu;this.width=field.width;this.height=field.height
    this.params={...params};this.buffers=buffers
    this.parameterBuffer=parameters;this.pipeline=pipeline;this.groups=groups
    this.iteration=field.iteration
  }
  static async create(field:GenesisField,
    params:GenesisParams=genesisDefaults):Promise<GenesisGpuSolver>{
    validateGenesis(field,params)
    const gpu=await requestStrictGpu('Handwired Neural Genesis')
    try{
      if(Math.ceil(field.width/8)>gpu.device.limits.maxComputeWorkgroupsPerDimension||
        Math.ceil(field.height/8)>gpu.device.limits.maxComputeWorkgroupsPerDimension)
        throw new Error('morphogen dispatch exceeds device limits')
      const bytes=field.state.byteLength
      const buffers:[GPUBuffer,GPUBuffer]=[
        gpu.allocate(bytes,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,'Morphogen state A'),
        gpu.allocate(bytes,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,'Morphogen state B'),
      ]
      gpu.device.queue.writeBuffer(buffers[0],0,field.state)
      const parameterBuffer=gpu.allocate(48,usage.UNIFORM|usage.COPY_DST,'Morphogen controls')
      const module=gpu.device.createShaderModule({code:genesisKernelWgsl})
      const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error')
      if(errors.length)throw new Error('Genesis WGSL: '+errors.map(x=>x.message).join('; '))
      const pipeline=await gpu.device.createComputePipelineAsync({
        layout:'auto',compute:{module,entryPoint:'main'},
      })
      const bind=(a:GPUBuffer,b:GPUBuffer)=>gpu.device.createBindGroup({
        layout:pipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:a}},{binding:1,resource:{buffer:b}},
          {binding:2,resource:{buffer:parameterBuffer}},
        ],
      })
      const engine=new GenesisGpuSolver(gpu,field,params,buffers,parameterBuffer,pipeline,
        [bind(buffers[0],buffers[1]),bind(buffers[1],buffers[0])])
      engine.setParameters(params)
      return engine
    }catch(error){gpu.dispose();throw error}
  }
  private check(){if(this.disposed)throw new Error('morphogen engine disposed');this.gpu.check()}
  get frame(){return this.iteration}
  get currentIndex(){return this.index}
  get currentBuffer(){return this.buffers[this.index]!}
  get numCells(){return this.width*this.height}
  setParameters(params:GenesisParams){
    this.check()
    if(![params.feed,params.kill,params.diffuseU,params.diffuseV].every(Number.isFinite)||
      params.feed<0||params.feed>.12||params.kill<0||params.kill>.12||
      params.diffuseU<0||params.diffuseU>.22||params.diffuseV<0||params.diffuseV>.22)
      throw new Error('invalid reaction coefficients')
    Object.assign(this.params,params)
    this.writeParameters()
  }
  setBrush(brush:GenesisBrush){
    this.check()
    if(![brush.x,brush.y,brush.radius].every(Number.isFinite)||brush.x<0||
      brush.x>1||brush.y<0||brush.y>1||brush.radius<0||brush.radius>100||
      ![0,1,2].includes(brush.mode))throw new Error('invalid morphogen brush')
    this.brush={...brush};this.writeParameters()
  }
  private writeParameters(){
    const bytes=new ArrayBuffer(48),view=new DataView(bytes)
    view.setUint32(0,this.width,true);view.setUint32(4,this.height,true)
    for(const [i,n] of [this.params.feed,this.params.kill,this.params.diffuseU,
      this.params.diffuseV,this.brush.x,this.brush.y].entries())
      view.setFloat32(8+i*4,n,true)
    view.setUint32(32,this.brush.mode,true);view.setFloat32(36,this.brush.radius,true)
    this.gpu.device.queue.writeBuffer(this.parameterBuffer,0,bytes)
  }
  encodeStep(encoder:GPUCommandEncoder){
    this.check()
    const pass=encoder.beginComputePass()
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[this.index]!)
    pass.dispatchWorkgroups(Math.ceil(this.width/8),Math.ceil(this.height/8))
    pass.end();this.index=1-this.index;this.iteration++
  }
  step(steps=1){
    this.check()
    if(!Number.isSafeInteger(steps)||steps<1||steps>64)throw new Error('invalid iteration batch')
    const encoder=this.gpu.device.createCommandEncoder()
    for(let i=0;i<steps;i++)this.encodeStep(encoder)
    this.gpu.device.queue.submit([encoder.finish()])
  }
  reset(field:GenesisField){
    this.check()
    if(field.width!==this.width||field.height!==this.height)throw new Error('morphogen grid mismatch')
    validateGenesis(field,this.params)
    this.gpu.device.queue.writeBuffer(this.buffers[0],0,field.state)
    this.index=0;this.iteration=field.iteration
  }
  async read():Promise<GenesisField>{
    this.check()
    const bytes=this.width*this.height*8
    const staging=this.gpu.device.createBuffer({size:bytes,usage:usage.COPY_DST|usage.MAP_READ})
    try{
      const encoder=this.gpu.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.currentBuffer,0,staging,0,bytes)
      this.gpu.device.queue.submit([encoder.finish()])
      await staging.mapAsync(1)
      const state=new Float32Array(staging.getMappedRange().slice(0))
      staging.unmap()
      this.check()
      return {width:this.width,height:this.height,state,iteration:this.iteration}
    }finally{staging.destroy()}
  }
  dispose(){if(this.disposed)return;this.disposed=true;this.gpu.dispose()}
}

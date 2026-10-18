import {requestStrictGpu,type GpuAllocation} from './frontiers-device.ts'
import {validateSpatialGrid,validateSpatialPoints,
  type SpatialGridConfig} from './frontiers-spatial.ts'

/**
 * Actual WebGPU linked-cell spatial index, with atomic cell insertion and
 * radius-filtered 27-cell neighborhood queries. All recurrent state is GPU
 * resident; readCounts() is an explicit synchronized verification readback.
 *
 * Insert ordering is nondeterministic but neighbor counts are not.
 * Requires query radius <= cell edge. Not a sorted hash grid, BVH or fluid solver.
 */
export const spatialGridWgsl = `
struct Params {
  origin:vec4<f32>,
  dims:vec4<u32>,
  physics:vec4<f32>,
  count:u32,padA:u32,padB:u32,padC:u32,
};
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> cellHeads:array<atomic<i32>>;
@group(0) @binding(2) var<storage,read_write> nextInCell:array<i32>;
@group(0) @binding(3) var<storage,read_write> neighborCount:array<u32>;
@group(0) @binding(4) var<uniform> params:Params;

fn cellCoordinates(pos:vec3<f32>)->vec3<i32>{
  return vec3<i32>(floor((pos-params.origin.xyz)/params.physics.x));
}
fn within(coord:vec3<i32>)->bool {
  return all(coord>=vec3<i32>(0)) &&
    coord.x<i32(params.dims.x)&&coord.y<i32(params.dims.y)&&coord.z<i32(params.dims.z);
}
fn linearCell(coord:vec3<i32>)->u32 {
  return (u32(coord.z)*params.dims.y+u32(coord.y))*params.dims.x+u32(coord.x);
}

@compute @workgroup_size(128)
fn clearCells(@builtin(global_invocation_id) id:vec3<u32>){
  if(id.x<params.dims.x*params.dims.y*params.dims.z){
    atomicStore(&cellHeads[id.x],-1);
  }
}
@compute @workgroup_size(128)
fn assignCells(@builtin(global_invocation_id) id:vec3<u32>){
  if(id.x>=params.count){return;}
  nextInCell[id.x]=-1;
  let coord=cellCoordinates(positions[id.x].xyz);
  if(!within(coord)){return;}
  let previous=atomicExchange(&cellHeads[linearCell(coord)],i32(id.x));
  nextInCell[id.x]=previous;
}
@compute @workgroup_size(128)
fn queryNeighbors(@builtin(global_invocation_id) id:vec3<u32>){
  if(id.x>=params.count){return;}
  let p=positions[id.x].xyz;
  let base=cellCoordinates(p);
  if(!within(base)){neighborCount[id.x]=0u;return;}
  var count=0u;
  for(var dz=-1;dz<=1;dz=dz+1){
    for(var dy=-1;dy<=1;dy=dy+1){
      for(var dx=-1;dx<=1;dx=dx+1){
        let coord=base+vec3<i32>(dx,dy,dz);
        if(!within(coord)){continue;}
        var node=atomicLoad(&cellHeads[linearCell(coord)]);
        var visited=0u;
        // Bound traversal even if a malformed list were ever constructed.
        loop{
          if(node<0||visited>=params.count){break;}
          let j=u32(node);
          if(j!=id.x){
            let delta=p-positions[j].xyz;
            if(dot(delta,delta)<=params.physics.z){count=count+1u;}
          }
          node=nextInCell[j];
          visited=visited+1u;
        }
      }
    }
  }
  neighborCount[id.x]=count;
}
`
const usages={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64,MAP_READ:1}
export class SpatialGridGpu {
  readonly gpu:GpuAllocation
  readonly config:SpatialGridConfig
  readonly pointCount:number
  readonly cellCount:number
  readonly positions:GPUBuffer
  readonly cellHeads:GPUBuffer
  readonly next:GPUBuffer
  readonly neighborCounts:GPUBuffer
  private readonly clear:GPUComputePipeline
  private readonly assign:GPUComputePipeline
  private readonly query:GPUComputePipeline
  private readonly bind:GPUBindGroup
  private disposed=false
  private frame=0
  private constructor(gpu:GpuAllocation,points:Float32Array,config:SpatialGridConfig,
    count:number,buffers:readonly [GPUBuffer,GPUBuffer,GPUBuffer,GPUBuffer,GPUBuffer],
    pipelines:readonly [GPUComputePipeline,GPUComputePipeline,GPUComputePipeline],
    bind:GPUBindGroup){
    this.gpu=gpu;this.config={...config,origin:[...config.origin] as [number,number,number],
      dimensions:[...config.dimensions] as [number,number,number]}
    this.pointCount=points.length/4;this.cellCount=count
    ;[this.positions,this.cellHeads,this.next,this.neighborCounts]=buffers
    ;[this.clear,this.assign,this.query]=pipelines
    this.bind=bind
  }
  static async create(points:Float32Array,config:SpatialGridConfig):Promise<SpatialGridGpu>{
    const cells=validateSpatialGrid(config),n=validateSpatialPoints(points)
    const gpu=await requestStrictGpu('Handwired GPU spatial grid')
    try{
      if(n>=2147483647)throw new Error('linked cell node index exceeds i32')
      const limit=gpu.device.limits.maxComputeWorkgroupsPerDimension*128
      if(n>limit||cells>limit)throw new Error('GPU dispatch limit exceeded')
      const alloc=(size:number,usage:number,label:string)=>
        gpu.allocate(size,usage,label)
      const input=alloc(points.byteLength,usages.STORAGE|usages.COPY_DST,'Spatial points xyz+w')
      const heads=alloc(cells*4,usages.STORAGE,'Atomic cell-head indices')
      const next=alloc(n*4,usages.STORAGE,'Intracell linked-list pointers')
      const counts=alloc(n*4,usages.STORAGE|usages.COPY_SRC,'Neighborhood cardinality')
      const uniform=alloc(64,usages.UNIFORM|usages.COPY_DST,'Spatial-grid shape/thresholds')
      gpu.device.queue.writeBuffer(input,0,points)
      const bytes=new ArrayBuffer(64),v=new DataView(bytes)
      for(let i=0;i<3;i++)v.setFloat32(i*4,config.origin[i]!,true)
      for(let i=0;i<3;i++)v.setUint32(16+i*4,config.dimensions[i]!,true)
      v.setFloat32(32,config.cellSize,true)
      v.setFloat32(36,config.radius,true)
      v.setFloat32(40,config.radius*config.radius,true)
      v.setUint32(48,n,true)
      gpu.device.queue.writeBuffer(uniform,0,bytes)
      const module=gpu.device.createShaderModule({code:spatialGridWgsl})
      const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error')
      if(errors.length)throw new Error('Spatial WGSL: '+errors.map(x=>x.message).join('; '))
      const layout=gpu.device.createBindGroupLayout({entries:[
        {binding:0,visibility:4,buffer:{type:'read-only-storage'}},
        {binding:1,visibility:4,buffer:{type:'storage'}},
        {binding:2,visibility:4,buffer:{type:'storage'}},
        {binding:3,visibility:4,buffer:{type:'storage'}},
        {binding:4,visibility:4,buffer:{type:'uniform'}},
      ]})
      const pipelineLayout=gpu.device.createPipelineLayout({bindGroupLayouts:[layout]})
      const pipeline=(entryPoint:string)=>gpu.device.createComputePipelineAsync({
        layout:pipelineLayout,compute:{module,entryPoint},
      })
      const [clear,assign,query]=await Promise.all([
        pipeline('clearCells'),pipeline('assignCells'),pipeline('queryNeighbors'),
      ])
      const bind=gpu.device.createBindGroup({layout,entries:[
        {binding:0,resource:{buffer:input}},
        {binding:1,resource:{buffer:heads}},
        {binding:2,resource:{buffer:next}},
        {binding:3,resource:{buffer:counts}},
        {binding:4,resource:{buffer:uniform}},
      ]})
      return new SpatialGridGpu(gpu,points,config,cells,
        [input,heads,next,counts,uniform],[clear,assign,query],bind)
    }catch(e){gpu.dispose();throw e}
  }
  private ensure(){if(this.disposed)throw new Error('spatial grid disposed');this.gpu.check()}
  get version(){return this.frame}
  uploadPoints(points:Float32Array){
    this.ensure()
    if(validateSpatialPoints(points)!==this.pointCount)
      throw new Error('cannot alter point count without rebuilding grid')
    this.gpu.device.queue.writeBuffer(this.positions,0,points)
  }
  /** Rebuild the 3D linked-cell adjacency without an unused neighbor-count pass.
   * Consumer physics kernels can immediately traverse these GPU-resident buffers
   * in later compute passes in the same command encoder.
   */
  encodeBuild(encoder:GPUCommandEncoder){
    this.ensure()
    const pass=encoder.beginComputePass({label:'GPU grid reset and atomic scatter'})
    pass.setBindGroup(0,this.bind)
    pass.setPipeline(this.clear)
    pass.dispatchWorkgroups(Math.ceil(this.cellCount/128))
    pass.setPipeline(this.assign)
    pass.dispatchWorkgroups(Math.ceil(this.pointCount/128))
    pass.end()
    this.frame++
  }
  /** Rebuild cells and compute the exact Euclidean neighbor counts. */
  encodeQuery(encoder:GPUCommandEncoder){
    this.encodeBuild(encoder)
    const pass=encoder.beginComputePass({label:'GPU 27-cell Euclidean radius count'})
    pass.setBindGroup(0,this.bind)
    pass.setPipeline(this.query)
    pass.dispatchWorkgroups(Math.ceil(this.pointCount/128))
    pass.end()
  }
  queryNow(){
    this.ensure()
    const encoder=this.gpu.device.createCommandEncoder()
    this.encodeQuery(encoder)
    this.gpu.device.queue.submit([encoder.finish()])
  }
  async readCounts():Promise<Uint32Array>{
    this.ensure()
    const size=this.pointCount*4
    const stage=this.gpu.device.createBuffer({size,usage:usages.COPY_DST|usages.MAP_READ})
    try{
      const encoder=this.gpu.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.neighborCounts,0,stage,0,size)
      this.gpu.device.queue.submit([encoder.finish()])
      await stage.mapAsync(1)
      const copy=new Uint32Array(stage.getMappedRange().slice(0))
      stage.unmap()
      this.ensure()
      return copy
    }finally{stage.destroy()}
  }
  async flush(){this.ensure();await this.gpu.device.queue.onSubmittedWorkDone();this.ensure()}
  dispose(){if(this.disposed)return;this.disposed=true;this.gpu.dispose()}
}

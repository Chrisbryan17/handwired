import {buildSwarmCircuit} from './model.ts'
import {emitInlineCircuitWgsl} from '../../src/runtime/webgpu-inline.ts'
import {initialPopulation, MAX_SWARM_AGENTS, type PointerState} from './simulation.ts'

/**
 * GPU-resident approximate Boids: O(agents * 49) neighborhood cell sampling,
 * two compute dispatches and instanced GPU drawing, no per-frame readback.
 *
 * This is deliberately a different (cell-aggregated) neighborhood algorithm
 * from the legacy exact pairwise CPU implementation; do not describe its
 * motion as exact numerical parity with legacy Boids.
 *
 * Each grid slot contains quantized atomic aggregates by species.
 * Quantization scale 65536 fits signed i32 at <=16384 agents and |v|<=0.85.
 */
export const GRID_DIMENSION = 24
export const GRID_RADIUS = 3
const CELL_COUNT = GRID_DIMENSION * GRID_DIMENSION * 3
const USAGE = {MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128,VERTEX:32} as const
const MAP_READ = 1
type Params = {count: number,mode: number,px:number,py:number,w:number,h:number}
const defs = `
struct Params {
  count:u32, mode:u32, pointerX:f32, pointerY:f32,
  renderWidth:f32,renderHeight:f32,step:f32,pad:f32,
};
struct Cell {
  count:atomic<i32>,
  sx:atomic<i32>,
  sy:atomic<i32>,
  svx:atomic<i32>,
  svy:atomic<i32>,
};
fn gridIndex(x:i32,y:i32,kind:u32)->u32 {
  return kind*576u+u32(y)*24u+u32(x);
}
fn boxCell(v:f32)->i32 {
  return clamp(i32(floor(v*24.0)),0,23);
}
fn bounded(x:f32,m:f32)->f32{return clamp(x,-m,m);}
`
const buckets = defs+`
@group(0) @binding(0) var<storage,read> agents:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> grid:array<Cell>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
  let i=tid.x;
  if(i>=p.count){return;}
  let s=agents[i];
  let index=gridIndex(boxCell(s.x),boxCell(s.y),i%3u);
  atomicAdd(&grid[index].count,1);
  atomicAdd(&grid[index].sx,i32(round(s.x*65536.0)));
  atomicAdd(&grid[index].sy,i32(round(s.y*65536.0)));
  atomicAdd(&grid[index].svx,i32(round(s.z*65536.0)));
  atomicAdd(&grid[index].svy,i32(round(s.w*65536.0)));
}`
const {circuit,outputWires}=buildSwarmCircuit()
const inline=emitInlineCircuitWgsl(circuit,outputWires,'handwiredBoid')
const integrate=defs+inline+`
@group(0) @binding(0) var<storage,read> agents:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> grid:array<Cell>;
@group(0) @binding(2) var<storage,read_write> nextAgents:array<vec4<f32>>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
  let i=tid.x;if(i>=p.count){return;}
  let state=agents[i];
  let x=state.x;let y=state.y;
  let vx=state.z;let vy=state.w;
  let gx=boxCell(x);let gy=boxCell(y);let species=i%3u;
  var seen:f32=0.0;
  var separation=vec2<f32>(0.0);
  var centers=vec2<f32>(0.0);
  var velocities=vec2<f32>(0.0);
  for(var oy=-3;oy<=3;oy=oy+1){
    let cy=gy+oy;if(cy<0||cy>=24){continue;}
    for(var ox=-3;ox<=3;ox=ox+1){
      let cx=gx+ox;if(cx<0||cx>=24){continue;}
      let cell=gridIndex(cx,cy,species);
      let count=atomicLoad(&grid[cell].count);
      if(count==0){continue;}
      var n=f32(count);
      var avg=vec2<f32>(
        f32(atomicLoad(&grid[cell].sx)),
        f32(atomicLoad(&grid[cell].sy)))/(n*65536.0);
      var avgV=vec2<f32>(
        f32(atomicLoad(&grid[cell].svx)),
        f32(atomicLoad(&grid[cell].svy)))/(n*65536.0);
      // Prevent self-interaction in the agent's own spatial cell.
      if(cx==gx&&cy==gy) {
        n=n-1.0;
        if(n<=0.0){continue;}
        avg=(avg*f32(count)-vec2<f32>(x,y))/n;
        avgV=(avgV*f32(count)-vec2<f32>(vx,vy))/n;
      }
      let displacement=vec2<f32>(x,y)-avg;
      let ds=dot(displacement,displacement);
      if(ds>0.0136){continue;}
      separation+=displacement*(0.065/(ds+0.002))*n;
      centers+=avg*n;velocities+=avgV*n;seen+=n;
    }
  }
  var source:array<f32,14>;
  source[0u]=x;source[1u]=y;source[2u]=vx;source[3u]=vy;
  for(var k=4u;k<14u;k=k+1u){source[k]=0.0;}
  if(seen>0.0){
    source[4u]=bounded(separation.x/seen,1.5);
    source[5u]=bounded(separation.y/seen,1.5);
    source[6u]=bounded((centers.x/seen-x)*6.0,1.0);
    source[7u]=bounded((centers.y/seen-y)*6.0,1.0);
    source[8u]=bounded((velocities.x/seen-vx)*2.0,1.0);
    source[9u]=bounded((velocities.y/seen-vy)*2.0,1.0);
  }
  if(p.mode!=0u){
    let d=vec2<f32>(p.pointerX-x,p.pointerY-y);
    let ds=dot(d,d);
    let sign=select(-1.0,1.0,p.mode==1u);
    let intensity=select(0.22/(ds+0.018),1.3,p.mode==1u);
    source[10u]=bounded(sign*d.x*intensity,1.8);
    source[11u]=bounded(sign*d.y*intensity,1.8);
  }
  source[12u]=select(select(0.0,(0.88-x)*12.0,x>0.88),(0.12-x)*12.0,x<0.12);
  source[13u]=select(select(0.0,(0.88-y)*12.0,y>0.88),(0.12-y)*12.0,y<0.12);
  let result=handwiredBoid(source);
  var point=vec2<f32>(result[0u],result[1u]);
  var velocity=vec2<f32>(result[2u],result[3u]);
  let speed=length(velocity);
  if(speed>0.85){velocity=velocity*(0.85/speed);}
  if(point.x<0.008||point.x>0.992){
    point.x=clamp(point.x,0.008,0.992);velocity.x=-velocity.x*0.6;
  }
  if(point.y<0.008||point.y>0.992){
    point.y=clamp(point.y,0.008,0.992);velocity.y=-velocity.y*0.6;
  }
  nextAgents[i]=vec4<f32>(point,velocity);
}`
const vertexWGSL=`
struct Params {
  count:u32, mode:u32, pointerX:f32, pointerY:f32,
  renderWidth:f32,renderHeight:f32,step:f32,pad:f32,
};
@group(0) @binding(0) var<storage,read> agents:array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p:Params;
struct VertexOutput{
  @builtin(position) position:vec4<f32>,
  @location(0) color:vec3<f32>,
  @location(1) @interpolate(flat) index:u32,
};
@vertex fn vs(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->VertexOutput {
  let agent=agents[i];
  var dir=agent.zw;
  let speed=length(dir);
  dir=select(vec2<f32>(1.0,0.0),dir/max(speed,0.00001),speed>0.00001);
  let side=vec2<f32>(-dir.y,dir.x);
  let tip=dir*3.8;
  let left=-dir*2.3+side*1.6;
  let right=-dir*2.3-side*1.6;
  let tail=-dir*7.2;
  var offset:vec2<f32>;
  switch(v) {
   case 0u:{offset=tip;}
   case 1u:{offset=left;}
   case 2u:{offset=right;}
   case 3u:{offset=left;}
   case 4u:{offset=right;}
   default:{offset=tail;}
  }
  let px=vec2<f32>(offset.x/max(1.0,p.renderWidth),offset.y/max(1.0,p.renderHeight));
  let center=vec2<f32>(agent.x*2.0-1.0,1.0-agent.y*2.0);
  let pos=center+vec2<f32>(px.x*2.0,-px.y*2.0);
  var result:VertexOutput;
  result.position=vec4<f32>(pos,0.0,1.0);
  let species=i%3u;
  result.color=select(select(vec3<f32>(0.46,0.96,0.82),vec3<f32>(0.96,0.72,0.52),species==1u),
    vec3<f32>(0.68,0.65,1.0),species==2u);
  result.index=v;
  return result;
}
@fragment fn fs(@location(0) color:vec3<f32>,@location(1) @interpolate(flat) index:u32)
  -> @location(0) vec4<f32> {
   let alpha=select(1.0,0.32,index>=3u);
   return vec4<f32>(color,alpha);
}`

export const swarmGpuShaders={buckets,integrate,vertexWGSL}
export class ResidentSwarmGpu {
  readonly count:number
  readonly device:GPUDevice
  private readonly context:GPUCanvasContext
  private readonly params:GPUBuffer
  private readonly positions:[GPUBuffer,GPUBuffer]
  private readonly cellGrid:GPUBuffer
  private readonly bucketPipeline:GPUComputePipeline
  private readonly stepPipeline:GPUComputePipeline
  private readonly renderPipeline:GPURenderPipeline
  private readonly buckets:[GPUBindGroup,GPUBindGroup]
  private readonly steps:[GPUBindGroup,GPUBindGroup]
  private readonly renders:[GPUBindGroup,GPUBindGroup]
  private current=0
  private disposed=false
  private lost:string|null=null
  private stepCounter=0
  private constructor(
    device:GPUDevice,context:GPUCanvasContext,count:number,
    params:GPUBuffer,positions:[GPUBuffer,GPUBuffer],cellGrid:GPUBuffer,
    pipelines:[GPUComputePipeline,GPUComputePipeline,GPURenderPipeline],
    groups:[GPUBindGroup[],GPUBindGroup[],GPUBindGroup[]],
  ){
    this.device=device;this.context=context;this.count=count;this.params=params
    this.positions=positions;this.cellGrid=cellGrid
    this.bucketPipeline=pipelines[0];this.stepPipeline=pipelines[1];this.renderPipeline=pipelines[2]
    this.buckets=groups[0] as [GPUBindGroup,GPUBindGroup]
    this.steps=groups[1] as [GPUBindGroup,GPUBindGroup]
    this.renders=groups[2] as [GPUBindGroup,GPUBindGroup]
    void device.lost.then(info=>{this.lost=info.message!==''?info.message:info.reason})
  }
  static async create(canvas:HTMLCanvasElement,count:number) {
    if(!Number.isSafeInteger(count)||count<1||count>MAX_SWARM_AGENTS)
      throw new Error('invalid swarm GPU population')
    const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'})
    if(!adapter)throw new Error('WebGPU adapter unavailable')
    const device=await adapter.requestDevice()
    const context=canvas.getContext('webgpu')
    if(!context){device.destroy();throw new Error('WebGPU canvas unavailable')}
    const format=navigator.gpu.getPreferredCanvasFormat()
    context.configure({device,format,alphaMode:'opaque'})
    const params=device.createBuffer({size:32,usage:USAGE.UNIFORM|USAGE.COPY_DST})
    const positions:[GPUBuffer,GPUBuffer]=[0,1].map(()=>device.createBuffer({
      size:count*16,usage:USAGE.STORAGE|USAGE.COPY_DST|USAGE.COPY_SRC,
    })) as [GPUBuffer,GPUBuffer]
    const cellGrid=device.createBuffer({size:CELL_COUNT*20,
      usage:USAGE.STORAGE|USAGE.COPY_DST})
    try{
      for(const [name,source] of Object.entries({buckets,integrate,vertexWGSL})){
        const shader=device.createShaderModule({code:source,label:'Handwired swarm '+name})
        const errors=(await shader.getCompilationInfo()).messages.filter(e=>e.type==='error')
        if(errors.length)throw new Error(name+' WGSL: '+errors.map(e=>
          e.lineNum+':'+e.linePos+' '+e.message).join('; '))
      }
      const bucketModule=device.createShaderModule({code:buckets})
      const integrateModule=device.createShaderModule({code:integrate})
      const drawModule=device.createShaderModule({code:vertexWGSL})
      const bucketPipeline=await device.createComputePipelineAsync({
        layout:'auto',compute:{module:bucketModule,entryPoint:'main'},
      })
      const stepPipeline=await device.createComputePipelineAsync({
        layout:'auto',compute:{module:integrateModule,entryPoint:'main'},
      })
      const renderPipeline=await device.createRenderPipelineAsync({
        layout:'auto',
        vertex:{module:drawModule,entryPoint:'vs'},
        fragment:{module:drawModule,entryPoint:'fs',targets:[{
          format,blend:{
            color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha',operation:'add'},
            alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha',operation:'add'},
          },
        }]},
        primitive:{topology:'triangle-list'},
      })
      const group=(pipeline:GPUComputePipeline|GPURenderPipeline,resources:GPUBuffer[])=>
        device.createBindGroup({layout:pipeline.getBindGroupLayout(0),
          entries:resources.map((buffer,index)=>({binding:index,resource:{buffer}}))})
      const bg=[group(bucketPipeline,[positions[0],cellGrid,params]),
        group(bucketPipeline,[positions[1],cellGrid,params])]
      const sg=[group(stepPipeline,[positions[0],cellGrid,positions[1],params]),
        group(stepPipeline,[positions[1],cellGrid,positions[0],params])]
      const rg=[group(renderPipeline,[positions[0],params]),
        group(renderPipeline,[positions[1],params])]
      const result=new ResidentSwarmGpu(device,context,count,params,positions,cellGrid,
        [bucketPipeline,stepPipeline,renderPipeline],[bg,sg,rg])
      result.reset()
      return result
    }catch(error){
      params.destroy();positions[0].destroy();positions[1].destroy();cellGrid.destroy()
      context.unconfigure();device.destroy();throw error
    }
  }
  reset(seed=3807){
    this.assertReady()
    const initial=initialPopulation(this.count,seed)
    this.device.queue.writeBuffer(this.positions[0],0,initial.state)
    this.current=0;this.stepCounter=0
  }
  private assertReady(){
    if(this.disposed)throw new Error('WebGPU swarm disposed')
    if(this.lost!==null)throw new Error('WebGPU device lost: '+this.lost)
  }
  frame(pointer:PointerState,renderWidth:number,renderHeight:number){
    this.assertReady()
    const data=new ArrayBuffer(32),v=new DataView(data)
    v.setUint32(0,this.count,true)
    v.setUint32(4,{off:0,attract:1,repel:2}[pointer.mode],true)
    v.setFloat32(8,pointer.x,true);v.setFloat32(12,pointer.y,true)
    v.setFloat32(16,renderWidth,true);v.setFloat32(20,renderHeight,true)
    v.setFloat32(24,this.stepCounter,true)
    this.device.queue.writeBuffer(this.params,0,data)
    const encoder=this.device.createCommandEncoder()
    encoder.clearBuffer(this.cellGrid)
    const bucket=encoder.beginComputePass()
    bucket.setPipeline(this.bucketPipeline)
    bucket.setBindGroup(0,this.buckets[this.current])
    bucket.dispatchWorkgroups(Math.ceil(this.count/128))
    bucket.end()
    const step=encoder.beginComputePass()
    step.setPipeline(this.stepPipeline)
    step.setBindGroup(0,this.steps[this.current])
    step.dispatchWorkgroups(Math.ceil(this.count/128))
    step.end()
    const next=1-this.current
    const pass=encoder.beginRenderPass({colorAttachments:[{
      view:this.context.getCurrentTexture().createView(),
      loadOp:'clear',storeOp:'store',clearValue:{r:0.022,g:0.033,b:0.075,a:1},
    }]})
    pass.setPipeline(this.renderPipeline)
    pass.setBindGroup(0,this.renders[next])
    pass.draw(6,this.count)
    pass.end()
    this.device.queue.submit([encoder.finish()])
    this.current=next;this.stepCounter++
  }
  async readState(){
    this.assertReady()
    const size=this.count*16
    const staging=this.device.createBuffer({size,usage:USAGE.MAP_READ|USAGE.COPY_DST})
    try{
      const cmd=this.device.createCommandEncoder()
      cmd.copyBufferToBuffer(this.positions[this.current],0,staging,0,size)
      this.device.queue.submit([cmd.finish()])
      await staging.mapAsync(MAP_READ)
      const state=new Float32Array(staging.getMappedRange().slice(0))
      staging.unmap();return state
    }finally{staging.destroy()}
  }
  destroy(){
    if(this.disposed)return
    this.disposed=true
    this.params.destroy();for(const b of this.positions)b.destroy()
    this.cellGrid.destroy();this.context.unconfigure();this.device.destroy()
  }
}

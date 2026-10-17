import {buildClothCircuit} from './model.ts'
import {emitInlineCircuitWgsl} from '../../src/runtime/webgpu-inline.ts'
import type {Cloth} from './physics.ts'

/** GPU resident 2-D spring/Verlet cloth with 8-connected per-vertex spring forces,
 * pinning, mouse constraints, persistent GPU tearing and Jacobi distance relaxation.
 * Every integration step executes the WGSL produced from the actual Handwired circuit.
 * No CPU calculation/readback is needed between simulation steps.
 */
export type ClothPointer = {
  dragged?:number|null, x?:number,y?:number,wind?:number,
  cut?:boolean,tearRadius?:number,
}
const Usage={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128}
const Mode={READ:1}
const params=`
struct Params {
  width:u32, height:u32, step:f32, wind:f32,
  dragged:i32, cut:u32, dragX:f32, dragY:f32,
  cutX:f32, cutY:f32, cutRadius:f32, restX:f32,
  restY:f32, p0:f32, p1:f32, p2:f32,
};
@group(0) @binding(0) var<uniform> p:Params;
`
const common=`
fn initial(i:u32)->vec2<f32> {
 let x=f32(i%p.width);let y=f32(i/p.width);
 return vec2<f32>(0.11+x*p.restX,0.08+y*p.restY);
}
fn directions() -> array<vec2<i32>,8> {
 return array<vec2<i32>,8>(
   vec2<i32>(-1,0),vec2<i32>(1,0),vec2<i32>(0,-1),vec2<i32>(0,1),
   vec2<i32>(-1,-1),vec2<i32>(1,-1),vec2<i32>(-1,1),vec2<i32>(1,1)
 );
}
fn isPinned(i:u32)->bool {
 let x=i%p.width;let y=i/p.width;
 return y==0u && (x==0u||x==p.width-1u||x%5u==0u);
}
fn isHeld(i:u32)->bool {return isPinned(i)||i32(i)==p.dragged;}
fn heldPosition(i:u32)->vec2<f32>{
 if(i32(i)==p.dragged){return vec2<f32>(p.dragX,p.dragY);}
 return initial(i);
}
fn valid(x:i32,y:i32)->bool {
 return x>=0&&y>=0&&x<i32(p.width)&&y<i32(p.height);
}
fn neighbor(i:u32,d:vec2<i32>)->u32 {
 let x=i32(i%p.width)+d.x; let y=i32(i/p.width)+d.y;
 return u32(y)*p.width+u32(x);
}
fn rest(d:vec2<i32>)->f32{
 return length(vec2<f32>(f32(d.x)*p.restX,f32(d.y)*p.restY));
}
`
const {circuit,outputWires}=buildClothCircuit()
const inline=emitInlineCircuitWgsl(circuit,outputWires,'handwiredVerlet')
const integrateWGSL=params+`
@group(0) @binding(1) var<storage,read> src:array<f32>;
@group(0) @binding(2) var<storage,read_write> dst:array<f32>;
@group(0) @binding(3) var<storage,read> masks:array<u32>;
`+common+inline+`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
 let i=tid.x;if(i>=p.width*p.height){return;}
 let a=i*4u;let current=vec2<f32>(src[a],src[a+1u]);
 var force=vec2<f32>(0.0);
 let gridX=i32(i%p.width);let gridY=i32(i/p.width);
 let d=directions();
 for(var k=0u;k<8u;k=k+1u) {
  if((masks[i]&(1u<<k))!=0u){continue;}
  let nx=gridX+d[k].x;let ny=gridY+d[k].y;
  if(!valid(nx,ny)){continue;}
  let n=neighbor(i,d[k])*4u;
  let delta=vec2<f32>(src[n],src[n+1u])-current;
  let distance=length(delta);
  if(distance<0.00000001){continue;}
  let tension=clamp((distance-rest(d[k]))*3.2,-0.6,0.6)/distance;
  force=force+delta*tension;
 }
 force=clamp(force,vec2<f32>(-2.0),vec2<f32>(2.0));
 var source:array<f32,8>;
 source[0u]=current.x;source[1u]=current.y;
 source[2u]=src[a+2u];source[3u]=src[a+3u];
 source[4u]=force.x;source[5u]=force.y;
 source[6u]=p.wind*sin(p.step*0.065+current.x*12.0+current.y*5.0)*0.65;
 source[7u]=0.10;
 let integrated=handwiredVerlet(source);
 var next=clamp(vec2<f32>(integrated[0u],integrated[1u]),
    vec2<f32>(0.012),vec2<f32>(0.988));
 if(isHeld(i)){next=heldPosition(i);}
 dst[a]=next.x;dst[a+1u]=next.y;
 dst[a+2u]=current.x;dst[a+3u]=current.y;
}`
const constraintWGSL=params+`
@group(0) @binding(1) var<storage,read> src:array<f32>;
@group(0) @binding(2) var<storage,read_write> dst:array<f32>;
@group(0) @binding(3) var<storage,read> masks:array<u32>;
`+common+`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
 let i=tid.x;if(i>=p.width*p.height){return;}
 let a=i*4u;
 var pos=vec2<f32>(src[a],src[a+1u]);
 if(isHeld(i)){pos=heldPosition(i);}
 else {
  let d=directions();
  let gx=i32(i%p.width);let gy=i32(i/p.width);
  var corrections=vec2<f32>(0.0);
  var total=0.0;
  for(var k=0u;k<8u;k=k+1u){
   if((masks[i]&(1u<<k))!=0u){continue;}
   let nx=gx+d[k].x;let ny=gy+d[k].y;
   if(!valid(nx,ny)){continue;}
   let j=neighbor(i,d[k]);let b=j*4u;
   let delta=vec2<f32>(src[b],src[b+1u])-pos;
   let dist=length(delta);
   if(dist<0.00000001){continue;}
   let magnitude=clamp((dist-rest(d[k]))/dist,-0.65,0.65);
   let held=select(1.0,2.0,isHeld(j));
   corrections=corrections+delta*magnitude*0.12*held;
   total=total+1.0;
  }
  pos=pos+corrections;
  let collision=pos-vec2<f32>(0.5,0.78);
  let radius=length(collision);
  if(radius<0.14&&radius>0.00000001){pos=vec2<f32>(0.5,0.78)+collision*(0.14/radius);}
  pos.x=clamp(pos.x,0.008,0.992);
  pos.y=clamp(pos.y,0.008,0.955);
 }
 dst[a]=pos.x;dst[a+1u]=pos.y;
 dst[a+2u]=src[a+2u];dst[a+3u]=src[a+3u];
}`
const tearWGSL=params+`
@group(0) @binding(1) var<storage,read> positions:array<f32>;
@group(0) @binding(2) var<storage,read> previous:array<u32>;
@group(0) @binding(3) var<storage,read_write> next:array<u32>;
`+common+`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
 let i=tid.x;if(i>=p.width*p.height){return;}
 var mask=previous[i];
 if(p.cut!=0u){
  let here=vec2<f32>(positions[i*4u],positions[i*4u+1u]);
  let d=directions();
  let gx=i32(i%p.width);let gy=i32(i/p.width);
  for(var k=0u;k<8u;k=k+1u){
   let nx=gx+d[k].x;let ny=gy+d[k].y;
   if(!valid(nx,ny)){continue;}
   let j=neighbor(i,d[k]);let there=vec2<f32>(positions[j*4u],positions[j*4u+1u]);
   let displacement=there-here;
   let center=vec2<f32>(p.cutX,p.cutY);
   let t=clamp(dot(center-here,displacement)/max(0.000000001,dot(displacement,displacement)),0.0,1.0);
   if(length(here+t*displacement-center)<p.cutRadius){mask=mask|(1u<<k);}
  }
 }
 next[i]=mask;
}`

export const clothGpuWgsl={integrateWGSL,constraintWGSL,tearWGSL}
export class ResidentClothGpu {
  readonly width:number
  readonly height:number
  readonly count:number
  readonly springs:number
  private readonly buffers:GPUBuffer[]
  private readonly state:[GPUBuffer,GPUBuffer]
  private readonly tear:[GPUBuffer,GPUBuffer]
  private readonly params:GPUBuffer
  private readonly integrate:GPUComputePipeline
  private readonly constrain:GPUComputePipeline
  private readonly tearPipeline:GPUComputePipeline
  private readonly integrations:[GPUBindGroup,GPUBindGroup]
  private readonly constraints:[GPUBindGroup,GPUBindGroup]
  private readonly tearGroups:[GPUBindGroup,GPUBindGroup]
  private stateParity=0
  private tearParity=0
  private disposed=false
  private loaded=false
  private lost:string|null=null
  private steps=0
  private constructor(private readonly device:GPUDevice,cloth:Cloth,buffers:GPUBuffer[],
    pipelines:[GPUComputePipeline,GPUComputePipeline,GPUComputePipeline],
    groups:[GPUBindGroup[],GPUBindGroup[],GPUBindGroup[]]){
    this.width=cloth.columns;this.height=cloth.rows;this.count=this.width*this.height
    this.springs=cloth.springs.length
    this.buffers=buffers
    this.params=buffers[0]!
    this.state=[buffers[1]!,buffers[2]!];this.tear=[buffers[3]!,buffers[4]!]
    this.integrate=pipelines[0];this.constrain=pipelines[1];this.tearPipeline=pipelines[2]
    this.integrations=groups[0] as [GPUBindGroup,GPUBindGroup]
    this.constraints=groups[1] as [GPUBindGroup,GPUBindGroup]
    this.tearGroups=groups[2] as [GPUBindGroup,GPUBindGroup]
    void device.lost.then(reason=>{this.lost=reason.message!==''?reason.message:reason.reason})
  }
  static async create(cloth:Cloth,device?:GPUDevice){
    const width=cloth.columns,height=cloth.rows,count=width*height
    if(count>4096||count<9||cloth.state.length!==count*4)throw new Error('invalid resident cloth mesh')
    const adapter=device||typeof navigator==='undefined'?null:await navigator.gpu.requestAdapter()
    const gpu=device??await adapter?.requestDevice()
    if(!gpu)throw new Error('WebGPU required for resident cloth')
    if(Math.ceil(count/64)>gpu.limits.maxComputeWorkgroupsPerDimension)throw new Error('cloth workgroups exceed GPU limits')
    const make=(size:number,usage:number)=>gpu.createBuffer({size,usage})
    const buffers=[
      make(64,Usage.UNIFORM|Usage.COPY_DST),
      make(count*16,Usage.STORAGE|Usage.COPY_SRC|Usage.COPY_DST),
      make(count*16,Usage.STORAGE|Usage.COPY_SRC|Usage.COPY_DST),
      make(count*4,Usage.STORAGE|Usage.COPY_DST|Usage.COPY_SRC),
      make(count*4,Usage.STORAGE|Usage.COPY_DST|Usage.COPY_SRC),
    ]
    try{
      const codes=[integrateWGSL,constraintWGSL,tearWGSL]
      const pipelines:GPUComputePipeline[]=[]
      for(let i=0;i<codes.length;i++){
        const shader=gpu.createShaderModule({code:codes[i]!,label:'Handwired resident cloth '+i})
        const errors=(await shader.getCompilationInfo()).messages.filter(e=>e.type==='error')
        if(errors.length)throw new Error('cloth WGSL '+i+': '+errors.map(e=>e.lineNum+':'+e.linePos+' '+e.message+' [ '+(codes[i]!.split(String.fromCharCode(10))[e.lineNum-1]??'')+' ]').join('; '))
        pipelines.push(await gpu.createComputePipelineAsync({
          layout:'auto',compute:{module:shader,entryPoint:'main'},
        }))
      }
      const group=(shader:GPUComputePipeline,src:GPUBuffer,dst:GPUBuffer,mask:GPUBuffer)=>
        gpu.createBindGroup({layout:shader.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:buffers[0]!}},
          {binding:1,resource:{buffer:src}},{binding:2,resource:{buffer:dst}},
          {binding:3,resource:{buffer:mask}},
        ]})
      const integration=[
        group(pipelines[0]!,buffers[1]!,buffers[2]!,buffers[3]!),
        group(pipelines[0]!,buffers[2]!,buffers[1]!,buffers[4]!)]
      const constraints=[
        group(pipelines[1]!,buffers[1]!,buffers[2]!,buffers[3]!),
        group(pipelines[1]!,buffers[2]!,buffers[1]!,buffers[3]!)]
      // Tear bindings are (positions, previous mask, next mask).
      const tearGroup=(previous:GPUBuffer,next:GPUBuffer)=>
        gpu.createBindGroup({layout:pipelines[2]!.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:buffers[0]!}},
          {binding:1,resource:{buffer:buffers[1]!}},
          {binding:2,resource:{buffer:previous}},
          {binding:3,resource:{buffer:next}},
        ]})
      const tearGroups=[tearGroup(buffers[3]!,buffers[4]!),tearGroup(buffers[4]!,buffers[3]!)]
      return new ResidentClothGpu(gpu,cloth,buffers,
        pipelines as [GPUComputePipeline,GPUComputePipeline,GPUComputePipeline],
        [integration,constraints,tearGroups])
    }catch(error){for(const buffer of buffers)buffer.destroy();throw error}
  }
  private ensure(){if(this.disposed)throw new Error('cloth GPU disposed')
    if(this.lost!==null)throw new Error('cloth GPU lost: '+this.lost)}
  load(cloth:Cloth){
    this.ensure()
    if(cloth.columns!==this.width||cloth.rows!==this.height)throw new Error('cloth dimensions mismatch')
    this.device.queue.writeBuffer(this.state[0],0,cloth.state)
    this.device.queue.writeBuffer(this.tear[0],0,new Uint32Array(this.count))
    this.device.queue.writeBuffer(this.tear[1],0,new Uint32Array(this.count))
    this.stateParity=0;this.tearParity=0;this.loaded=true;this.steps=0
  }
  private setParams(interaction:ClothPointer={}){
    const bytes=new ArrayBuffer(64),v=new DataView(bytes)
    v.setUint32(0,this.width,true);v.setUint32(4,this.height,true)
    v.setFloat32(8,this.steps,true);v.setFloat32(12,interaction.wind??1.2,true)
    v.setInt32(16,interaction.dragged??-1,true)
    v.setUint32(20,interaction.cut===true?1:0,true)
    v.setFloat32(24,interaction.x??.5,true);v.setFloat32(28,interaction.y??.5,true)
    v.setFloat32(32,interaction.x??.5,true);v.setFloat32(36,interaction.y??.5,true)
    v.setFloat32(40,interaction.tearRadius??.035,true)
    v.setFloat32(44,.78/(this.width-1),true)
    v.setFloat32(48,.59/(this.height-1),true)
    this.device.queue.writeBuffer(this.params,0,bytes)
  }
  async step(interaction:ClothPointer={}){
    this.ensure();if(!this.loaded)throw new Error('load cloth before stepping')
    this.setParams(interaction)
    const encoder=this.device.createCommandEncoder()
    const dispatch=(pipeline:GPUComputePipeline,group:GPUBindGroup)=>{
      const pass=encoder.beginComputePass()
      pass.setPipeline(pipeline);pass.setBindGroup(0,group)
      pass.dispatchWorkgroups(Math.ceil(this.count/64))
      pass.end()
    }
    // Tear happens on-device and is persistent, before the spring integration.
    if(interaction.cut===true){
      dispatch(this.tearPipeline,this.tearGroups[this.tearParity])
      this.tearParity=1-this.tearParity
    }
    // The state ping-pong returns to its starting buffer after one integration
    // and three constraint passes. Keep the initial state buffer persistent.
    const a=this.stateParity
    // Groups were compiled with the first tear mask. When tearing, masks cannot be
    // read through that bind group; the integration/constraint passes are rebuilt
    // with the current mask for each pass.
    const mask=this.tear[this.tearParity]
    const create=(pipeline:GPUComputePipeline,src:GPUBuffer,dst:GPUBuffer)=>{
      return this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:this.params}},
        {binding:1,resource:{buffer:src}},
        {binding:2,resource:{buffer:dst}},
        {binding:3,resource:{buffer:mask}},
      ]})
    }
    dispatch(this.integrate,create(this.integrate,this.state[a],this.state[1-a]))
    let parity=1-a
    for(let iteration=0;iteration<3;iteration++){
      dispatch(this.constrain,create(this.constrain,this.state[parity],this.state[1-parity]))
      parity=1-parity
    }
    this.device.queue.submit([encoder.finish()])
    await this.device.queue.onSubmittedWorkDone()
    this.stateParity=parity;this.steps++;this.ensure()
  }
  async read():Promise<Float32Array>{
    this.ensure();if(!this.loaded)throw new Error('load cloth before reading')
    const size=this.count*16,staging=this.device.createBuffer({size,usage:Usage.COPY_DST|Usage.MAP_READ})
    try{
      const encoder=this.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.state[this.stateParity],0,staging,0,size)
      this.device.queue.submit([encoder.finish()])
      await staging.mapAsync(Mode.READ)
      const values=new Float32Array(staging.getMappedRange().slice(0))
      staging.unmap();return values
    }finally{staging.destroy()}
  }
  async readTearMasks():Promise<Uint32Array>{
    this.ensure()
    if(!this.loaded)throw new Error('load cloth before reading tear masks')
    const size=this.count*4
    const staging=this.device.createBuffer({size,usage:Usage.COPY_DST|Usage.MAP_READ})
    try{
      const encoder=this.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.tear[this.tearParity],0,staging,0,size)
      this.device.queue.submit([encoder.finish()])
      await staging.mapAsync(Mode.READ)
      const masks=new Uint32Array(staging.getMappedRange().slice(0))
      staging.unmap();return masks
    }finally{staging.destroy()}
  }
  destroy(){if(this.disposed)return;this.disposed=true;for(const buffer of this.buffers)buffer.destroy()}
}

import {buildFluidCircuit, FLUID_CHANNELS} from './model.ts'
import {emitInlineCircuitWgsl} from '../../src/runtime/webgpu-inline.ts'
import type {Fluid} from './physics.ts'

/**
 * Fully GPU-resident 2D stable-fluid core.
 * The 5-channel advection kernel INLINES an emitted Handwired circuit.
 * Diffusion, divergence, Jacobi pressure and projection are separate WebGPU
 * stencil passes. Host readback is optional, never required between steps.
 */
const BufferUsage = {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128}
const MapMode = {READ: 1}
export type FluidPointer = {x:number,y:number,dx:number,dy:number,strength:number}
const uniform = `
struct Params {
  width:u32, height:u32, dt:f32, strength:f32,
  mouseX:f32, mouseY:f32, dx:f32, dy:f32,
  pointerActive:u32, phase:f32, p0:u32, p1:u32,
};
@group(0) @binding(0) var<uniform> p:Params;
`
const index = `
fn cell(x:i32,y:i32)->u32 {
  let xx=u32(clamp(x,0,i32(p.width)-1));
  let yy=u32(clamp(y,0,i32(p.height)-1));
  return yy*p.width+xx;
}
fn fieldAt(x:i32,y:i32,c:u32)->f32 {return src[cell(x,y)*5u+c];}
`
const gid = '@builtin(global_invocation_id) gid:vec3<u32>'
const header = '@compute @workgroup_size(8,8) fn main(' + gid + ') {\n' +
  'if(gid.x>=p.width||gid.y>=p.height){return;}\n' +
  'let x=i32(gid.x);let y=i32(gid.y);let i=gid.y*p.width+gid.x;\n'
const sample = '@group(0) @binding(1) var<storage,read> src:array<f32>;\n' +
  '@group(0) @binding(2) var<storage,read_write> dst:array<f32>;\n'

const {circuit,outputWires}=buildFluidCircuit()
const inline = emitInlineCircuitWgsl(circuit,outputWires,'handwiredAdvection')
const handwrittenAdvection = `
fn handwiredAdvection(sourceCells:array<f32,22>) -> array<f32,5> {
  var result:array<f32,5>;
  let tx=sourceCells[20u];let ty=sourceCells[21u];
  for(var k=0u;k<5u;k=k+1u){
    let base=k*4u;
    let top=sourceCells[base]+(sourceCells[base+1u]-sourceCells[base])*tx;
    let bottom=sourceCells[base+2u]+(sourceCells[base+3u]-sourceCells[base+2u])*tx;
    result[k]=top+(bottom-top)*ty;
  }
  return result;
}
`
const advectionWGSL = uniform + sample + index + inline + header + `
let u=fieldAt(x,y,0u);
let v=fieldAt(x,y,1u);
let bx=clamp(f32(x)-u*p.dt,0.0,f32(p.width)-1.000001);
let by=clamp(f32(y)-v*p.dt,0.0,f32(p.height)-1.000001);
let x0=i32(floor(bx));let y0=i32(floor(by));
let x1=min(x0+1,i32(p.width)-1);let y1=min(y0+1,i32(p.height)-1);
var sources:array<f32,22>;
for(var k=0u;k<5u;k=k+1u){
  sources[k*4u]=fieldAt(x0,y0,k);
  sources[k*4u+1u]=fieldAt(x1,y0,k);
  sources[k*4u+2u]=fieldAt(x0,y1,k);
  sources[k*4u+3u]=fieldAt(x1,y1,k);
}
sources[20u]=bx-f32(x0);sources[21u]=by-f32(y0);
let interpolated=handwiredAdvection(sources);
var impulse=0.0;
if(p.pointerActive==1u){
  let distance=vec2<f32>(f32(x)-p.mouseX*f32(p.width),f32(y)-p.mouseY*f32(p.height));
  impulse=exp(-dot(distance,distance)/9.0)*p.strength;
}
dst[i*5u]=clamp(interpolated[0u]+p.dx*impulse*8.0,-3.0,3.0);
dst[i*5u+1u]=clamp(interpolated[1u]+p.dy*impulse*8.0,-3.0,3.0);
for(var k=2u;k<5u;k=k+1u){
  let color=0.35+0.6*(1.0+sin(p.phase+f32(k-2u)*2.1));
  dst[i*5u+k]=clamp(interpolated[k]+impulse*color,0.0,6.0);
}
}`
const diffusionWGSL = uniform + sample + index + header + `
for(var k=0u;k<5u;k=k+1u){
 let center=src[i*5u+k];
 if(x==0||y==0||x==i32(p.width)-1||y==i32(p.height)-1){
  dst[i*5u+k]=center; continue;
 }
 let neighbors=0.25*(fieldAt(x-1,y,k)+fieldAt(x+1,y,k)+fieldAt(x,y-1,k)+fieldAt(x,y+1,k));
 let strength=select(0.016,0.006,k>=2u);
 dst[i*5u+k]=clamp(center+(neighbors-center)*strength,select(-3.0,0.0,k>=2u),select(3.0,6.0,k>=2u));
}
}`
const divergenceWGSL = uniform +
  '@group(0) @binding(1) var<storage,read> src:array<f32>;\n' +
  '@group(0) @binding(2) var<storage,read_write> dst:array<f32>;\n' +
  index + header + `
if(x==0||y==0||x==i32(p.width)-1||y==i32(p.height)-1){dst[i]=0.0;return;}
dst[i]=-0.5*(fieldAt(x+1,y,0u)-fieldAt(x-1,y,0u)+fieldAt(x,y+1,1u)-fieldAt(x,y-1,1u));
}`
const jacobiWGSL = uniform + sample +
  '@group(0) @binding(3) var<storage,read> divergence:array<f32>;\n' +
  `fn pressureAt(x:i32,y:i32)->f32 {
     let xx=u32(clamp(x,0,i32(p.width)-1));let yy=u32(clamp(y,0,i32(p.height)-1));
     return src[yy*p.width+xx];
   }\n` + header + `
if(x==0||y==0||x==i32(p.width)-1||y==i32(p.height)-1){dst[i]=0.0;return;}
dst[i]=0.25*(pressureAt(x-1,y)+pressureAt(x+1,y)+pressureAt(x,y-1)+pressureAt(x,y+1)+divergence[i]);
}`
const projectWGSL = uniform + sample + index +
  '@group(0) @binding(3) var<storage,read> pressure:array<f32>;\n' +
  `fn pressureAt(x:i32,y:i32)->f32 {
    let xx=u32(clamp(x,0,i32(p.width)-1));let yy=u32(clamp(y,0,i32(p.height)-1));
    return pressure[yy*p.width+xx];
  }\n` + header + `
let boundary=x==0||y==0||x==i32(p.width)-1||y==i32(p.height)-1;
let ux=src[i*5u];let uy=src[i*5u+1u];
dst[i*5u]=select(clamp(ux-0.5*(pressureAt(x+1,y)-pressureAt(x-1,y)),-3.0,3.0),0.0,boundary);
dst[i*5u+1u]=select(clamp(uy-0.5*(pressureAt(x,y+1)-pressureAt(x,y-1)),-3.0,3.0),0.0,boundary);
for(var k=2u;k<5u;k=k+1u){dst[i*5u+k]=clamp(src[i*5u+k]*0.998,0.0,6.0);}
}`

export const fluidGpuWgsl = {advectionWGSL,diffusionWGSL,divergenceWGSL,jacobiWGSL,projectWGSL}

const fluidDisplayWgsl = `
@group(0) @binding(0) var<storage,read> field:array<f32>;
struct Geometry { @builtin(position) pos:vec4<f32>,@location(0) uv:vec2<f32> }
@vertex fn vs(@builtin(vertex_index) vertex:u32)->Geometry {
  var pos=array<vec2<f32>,3>(
    vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
  var out:Geometry;
  out.pos=vec4<f32>(pos[vertex],0.0,1.0);
  out.uv=vec2<f32>(0.5+0.5*pos[vertex].x,0.5-0.5*pos[vertex].y);
  return out;
}
struct Dimensions { width:u32,height:u32,pad0:u32,pad1:u32 };
@group(0) @binding(1) var<uniform> dims:Dimensions;
@fragment fn fs(@location(0) uv:vec2<f32>)->@location(0) vec4<f32> {
  let pixel=clamp(uv*vec2<f32>(f32(dims.width),f32(dims.height))-vec2<f32>(0.5),
    vec2<f32>(0.0),vec2<f32>(f32(dims.width-1u),f32(dims.height-1u)));
  let x0=u32(floor(pixel.x));let y0=u32(floor(pixel.y));
  let x1=min(x0+1u,dims.width-1u);let y1=min(y0+1u,dims.height-1u);
  let dx=pixel.x-f32(x0);let dy=pixel.y-f32(y0);
  let a=(y0*dims.width+x0)*5u;
  let b=(y0*dims.width+x1)*5u;
  let c=(y1*dims.width+x0)*5u;
  let d=(y1*dims.width+x1)*5u;
  let top=mix(vec3<f32>(field[a+2u],field[a+3u],field[a+4u]),
    vec3<f32>(field[b+2u],field[b+3u],field[b+4u]),dx);
  let bottom=mix(vec3<f32>(field[c+2u],field[c+3u],field[c+4u]),
    vec3<f32>(field[d+2u],field[d+3u],field[d+4u]),dx);
  let color=mix(top,bottom,dy);
  let rgb=vec3<f32>(1.0)-exp(-max(color,vec3<f32>(0.0))*1.15);
  let glow=min(1.0,(rgb.x+rgb.y+rgb.z)*0.22);
  return vec4<f32>(vec3<f32>(0.0235,0.0235,0.0863)+
    rgb*vec3<f32>(0.92,0.82,0.86)+glow*vec3<f32>(0.15,0.13,0.16),1.0);
}
`

export type FluidAdvectionKernel = 'handwired' | 'handwritten'

export class ResidentFluidGpu {
  readonly width:number
  readonly height:number
  readonly count:number
  readonly adapterInfo:string
  private readonly buffers:GPUBuffer[]=[]
  private readonly fields:[GPUBuffer,GPUBuffer]
  private readonly pressure:[GPUBuffer,GPUBuffer]
  private readonly divergence:GPUBuffer
  private readonly params:GPUBuffer
  private readonly pipelines:Record<string,GPUComputePipeline>
  private readonly groups:Record<string,GPUBindGroup[]>
  private displayContext:GPUCanvasContext|null=null
  private displayPipeline:GPURenderPipeline|null=null
  private displayGroups:[GPUBindGroup,GPUBindGroup]|null=null
  private displayDims:GPUBuffer|null=null
  private current=0
  private loaded=false
  private lost:string|null=null
  private disposed=false
  private stepCount=0
  private constructor(private readonly device:GPUDevice,width:number,height:number,
    buffers:GPUBuffer[],pipelines:Record<string,GPUComputePipeline>,
    groups:Record<string,GPUBindGroup[]>,adapterInfo:string){
    this.width=width;this.height=height;this.count=width*height
    this.buffers=buffers;this.fields=[buffers[1]!,buffers[2]!]
    this.pressure=[buffers[3]!,buffers[4]!];this.divergence=buffers[5]!
    this.params=buffers[0]!;this.pipelines=pipelines;this.groups=groups
    this.adapterInfo=adapterInfo
    void device.lost.then(reason=>{this.lost=reason.message !== '' ? reason.message : reason.reason})
  }
  static async create(width:number,height:number,provided?:GPUDevice,kernel:FluidAdvectionKernel='handwired') {
    if(!Number.isInteger(width)||!Number.isInteger(height)||width<8||height<8||
      width*height>65536)throw new Error('invalid resident fluid dimensions')
    const adapter=provided || typeof navigator === 'undefined' ? null : await navigator.gpu.requestAdapter()
    const device=provided??await adapter?.requestDevice()
    if(!device)throw new Error('real WebGPU device required for resident fluid solver')
    const bytes=width*height*FLUID_CHANNELS*4,scalar=width*height*4
    for(const size of [bytes,scalar]){
      if(size>device.limits.maxStorageBufferBindingSize||size>device.limits.maxBufferSize)
        throw new Error('fluid buffers exceed device storage limits')
    }
    if(Math.ceil(width/8)>device.limits.maxComputeWorkgroupsPerDimension||
      Math.ceil(height/8)>device.limits.maxComputeWorkgroupsPerDimension)
      throw new Error('fluid grid exceeds device dispatch limits')
    const make=(size:number,usage:number)=>device.createBuffer({size,usage})
    const buffers=[
      make(48,BufferUsage.UNIFORM|BufferUsage.COPY_DST),
      make(bytes,BufferUsage.STORAGE|BufferUsage.COPY_DST|BufferUsage.COPY_SRC),
      make(bytes,BufferUsage.STORAGE|BufferUsage.COPY_DST|BufferUsage.COPY_SRC),
      make(scalar,BufferUsage.STORAGE|BufferUsage.COPY_DST),
      make(scalar,BufferUsage.STORAGE|BufferUsage.COPY_DST),
      make(scalar,BufferUsage.STORAGE|BufferUsage.COPY_DST),
    ]
    try{
      const shaders={advect:kernel==='handwired'?advectionWGSL:advectionWGSL.replace(inline,handwrittenAdvection),diffuse:diffusionWGSL,
        divergence:divergenceWGSL,jacobi:jacobiWGSL,project:projectWGSL}
      const pipelines:Record<string,GPUComputePipeline>={}
      for(const [name,source] of Object.entries(shaders)){
        const module=device.createShaderModule({code:source,label:'Handwired resident fluid '+name})
        const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error')
        if(errors.length)throw new Error(name+' WGSL: '+errors.map(x=>x.message).join('; '))
        pipelines[name]=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}})
      }
      const groups:Record<string,GPUBindGroup[]>={}
      const group=(name:string,src:GPUBuffer,dst:GPUBuffer,aux?:GPUBuffer)=>{
        const entries:GPUBindGroupEntry[]=[
          {binding:0,resource:{buffer:buffers[0]!}},
          {binding:1,resource:{buffer:src}},
          {binding:2,resource:{buffer:dst}},
        ]
        if(aux)entries.push({binding:3,resource:{buffer:aux}})
        return device.createBindGroup({layout:pipelines[name]!.getBindGroupLayout(0),entries})
      }
      groups['advect']=[group('advect',buffers[1]!,buffers[2]!),group('advect',buffers[2]!,buffers[1]!)]
      groups['diffuse']=[group('diffuse',buffers[2]!,buffers[1]!),group('diffuse',buffers[1]!,buffers[2]!)]
      groups['divergence']=[group('divergence',buffers[1]!,buffers[5]!),group('divergence',buffers[2]!,buffers[5]!)]
      groups['jacobi']=[group('jacobi',buffers[3]!,buffers[4]!,buffers[5]),
        group('jacobi',buffers[4]!,buffers[3]!,buffers[5])]
      groups['project']=[]
      for(let parity=0;parity<2;parity++){
        for(let pressureIndex=0;pressureIndex<2;pressureIndex++){
          groups['project'].push(group('project',buffers[1+parity]!,
            buffers[2-parity],buffers[3+pressureIndex]))
        }
      }
      const label=adapter?JSON.stringify(adapter.info):'user-supplied GPUDevice'
      return new ResidentFluidGpu(device,width,height,buffers,pipelines,groups,label)
    }catch(error){for(const b of buffers)b.destroy();throw error}
  }
  private ensure(){if(this.disposed)throw new Error('fluid GPU destroyed')
    if(this.lost !== null)throw new Error('fluid GPU device lost: '+this.lost)}
  load(fluid:Fluid|Float32Array) {
    this.ensure()
    const fields=fluid instanceof Float32Array?fluid:fluid.fields
    if(fields.length!==this.count*5||!fields.every(Number.isFinite))
      throw new Error('invalid initial fluid field')
    this.device.queue.writeBuffer(this.fields[0],0,fields)
    this.loaded=true;this.current=0;this.stepCount=0
  }
  private writeParams(pointer?:FluidPointer){
    const params=new ArrayBuffer(48),v=new DataView(params)
    v.setUint32(0,this.width,true);v.setUint32(4,this.height,true)
    v.setFloat32(8,.48,true);v.setFloat32(12,pointer?.strength??0,true)
    v.setFloat32(16,pointer?.x??.5,true);v.setFloat32(20,pointer?.y??.5,true)
    v.setFloat32(24,pointer?.dx??0,true);v.setFloat32(28,pointer?.dy??0,true)
    v.setUint32(32,pointer?1:0,true);v.setFloat32(36,this.stepCount*.035,true)
    this.device.queue.writeBuffer(this.params,0,params)
  }
  async attachCanvas(canvas:HTMLCanvasElement){
    this.ensure()
    const context=canvas.getContext('webgpu') as GPUCanvasContext|null
    if(!context)throw new Error('WebGPU rendering context unavailable')
    const format=navigator.gpu.getPreferredCanvasFormat()
    context.configure({device:this.device,format,alphaMode:'opaque'})
    const dimensions=this.device.createBuffer({size:16,usage:BufferUsage.UNIFORM|BufferUsage.COPY_DST})
    this.device.queue.writeBuffer(dimensions,0,new Uint32Array([this.width,this.height,0,0]))
    try{
      const shader=this.device.createShaderModule({code:fluidDisplayWgsl})
      const errors=(await shader.getCompilationInfo()).messages.filter(m=>m.type==='error')
      if(errors.length)throw new Error('fluid display WGSL: '+errors.map(m=>m.message).join('; '))
      const pipeline=await this.device.createRenderPipelineAsync({
        layout:'auto',vertex:{module:shader,entryPoint:'vs'},
        fragment:{module:shader,entryPoint:'fs',targets:[{format}]},
        primitive:{topology:'triangle-list'},
      })
      const bind=(buffer:GPUBuffer)=>this.device.createBindGroup({
        layout:pipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer}},{binding:1,resource:{buffer:dimensions}},
        ],
      })
      this.displayContext=context;this.displayPipeline=pipeline
      this.displayGroups=[bind(this.fields[0]),bind(this.fields[1])]
      this.displayDims=dimensions
    }catch(error){dimensions.destroy();context.unconfigure();throw error}
  }
  frame(pointer?:FluidPointer){

    this.ensure()
    if(!this.loaded)throw new Error('load fluid before stepping')
    this.writeParams(pointer)
    const encoder=this.device.createCommandEncoder()
    const pass=(name:string,index:number)=>{
      const run=encoder.beginComputePass()
      run.setPipeline(this.pipelines[name]!)
      run.setBindGroup(0,this.groups[name]![index]!)
      run.dispatchWorkgroups(Math.ceil(this.width/8),Math.ceil(this.height/8))
      run.end()
    }
    const p=this.current
    pass('advect',p)
    pass('diffuse',p)
    pass('divergence',p)
    encoder.clearBuffer(this.pressure[0]);encoder.clearBuffer(this.pressure[1])
    for(let n=0;n<20;n++)pass('jacobi',n%2)
    pass('project',p*2)
    const next=1-p
    if(this.displayContext!==null&&this.displayPipeline!==null&&this.displayGroups!==null){
      const render=encoder.beginRenderPass({colorAttachments:[{
        view:this.displayContext.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',
        clearValue:{r:0.02,g:0.02,b:0.065,a:1},
      }]})
      render.setPipeline(this.displayPipeline)
      render.setBindGroup(0,this.displayGroups[next])
      render.draw(3)
      render.end()
    }
    this.device.queue.submit([encoder.finish()])
    this.current=next;this.stepCount++;this.ensure()
  }
  async step(pointer?:FluidPointer){
    this.frame(pointer)
    await this.device.queue.onSubmittedWorkDone()
    this.ensure()
  }
  async read():Promise<Float32Array> {
    this.ensure()
    if(!this.loaded)throw new Error('load fluid before reading')
    const bytes=this.count*FLUID_CHANNELS*4
    const staging=this.device.createBuffer({size:bytes,usage:BufferUsage.MAP_READ|BufferUsage.COPY_DST})
    try{
      const encoder=this.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.fields[this.current],0,staging,0,bytes)
      this.device.queue.submit([encoder.finish()])
      await staging.mapAsync(MapMode.READ)
      const result=new Float32Array(staging.getMappedRange().slice(0))
      staging.unmap()
      return result
    }finally{staging.destroy()}
  }
  destroy(){if(this.disposed)return;this.disposed=true;this.displayDims?.destroy();this.displayContext?.unconfigure();for(const b of this.buffers)b.destroy()}
}

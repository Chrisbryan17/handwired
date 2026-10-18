import {emitInlineCircuitWgsl} from '../../../src/runtime/webgpu-inline.ts'
import {requestStrictGpu,type GpuAllocation} from '../../../src/runtime/frontiers-device.ts'
import {buildLightThroughputCircuit} from './model.ts'

/** Brute-force primitive, physically based progressive Monte Carlo path tracing.
 * Non-cosmetic: traces primary+secondary rays, evaluates area-light visibility,
 * cosine-weighted Lambertian BRDF, perfect mirrors, dielectric transmission
 * and Russian roulette. Each bounce multiplies spectral RGB throughput using
 * a function emitted from a real Handwired Circuit.
 *
 * GPU state is one per-pixel 4-channel accumulation buffer, never uploaded or
 * read back between frames. BVH, GGX, MIS and denoising remain open milestones.
 */
const circuit=buildLightThroughputCircuit()
export const lightKernelWgsl=`
struct Params {
 width:u32,height:u32,sampleIndex:u32,maxBounces:u32,
 yaw:f32,pitch:f32,exposure:f32,lightStrength:f32,
 ior:f32,pad0:f32,pad1:u32,pad2:u32,
};
struct Ray {origin:vec3<f32>,direction:vec3<f32>};
struct Hit {
 t:f32,position:vec3<f32>,normal:vec3<f32>,
 color:vec3<f32>,material:u32,emission:vec3<f32>,
 frontFace:bool,
};
@group(0) @binding(0) var<storage,read_write> accumulation:array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params:Params;
const E:f32=0.0008;
const PI:f32=3.141592653589793;
fn rng(state:ptr<function,u32>)->f32 {
 var x=(*state);
 x=x^(x<<13u);x=x^(x>>17u);x=x^(x<<5u);
 (*state)=x;
 return f32(x&0x00ffffffu)*(1.0/16777216.0);
}
fn miss(maxT:f32)->Hit {
 return Hit(maxT,vec3<f32>(0.0),vec3<f32>(0.0),
  vec3<f32>(0.0),9u,vec3<f32>(0.0),false);
}
fn sphere(ray:Ray,center:vec3<f32>,radius:f32,color:vec3<f32>,
 kind:u32,result:ptr<function,Hit>) {
 let oc=ray.origin-center;
 let b=dot(oc,ray.direction);
 let determinant=b*b-dot(oc,oc)+radius*radius;
 if(determinant<0.0){return;}
 let root=sqrt(determinant);
 var t=-b-root;
 if(t<E||t>=(*result).t){t=-b+root;}
 if(t<E||t>=(*result).t){return;}
 let position=ray.origin+ray.direction*t;
 let outward=normalize(position-center);
 let front=dot(ray.direction,outward)<0.0;
 (*result).t=t;
 (*result).position=position;
 (*result).normal=select(-outward,outward,front);
 (*result).color=color;
 (*result).material=kind;
 (*result).emission=vec3<f32>(0.0);
 (*result).frontFace=front;
}
fn wall(ray:Ray,axis:u32,distance:f32,normal:vec3<f32>,
 color:vec3<f32>,result:ptr<function,Hit>) {
 let d=ray.direction[axis];
 if(abs(d)<0.00000001){return;}
 let t=(distance-ray.origin[axis])/d;
 if(t<E||t>=(*result).t){return;}
 let pos=ray.origin+ray.direction*t;
 if(abs(pos.x)>1.7208||abs(pos.y)>1.3708||
   pos.z< -4.6608||pos.z>1.2){return;}
 let front=dot(ray.direction,normal)<0.0;
 (*result).t=t;(*result).position=pos;
 (*result).normal=select(-normal,normal,front);
 (*result).color=color;(*result).material=0u;
 (*result).emission=vec3<f32>(0.0);(*result).frontFace=front;
}
fn traceScene(ray:Ray,maxT:f32)->Hit{
 var hit=miss(maxT);
 sphere(ray,vec3<f32>(-.70,-.61,-2.92),.74,
  vec3<f32>(.96,.97,.99),1u,&hit);
 sphere(ray,vec3<f32>(.74,-.74,-3.18),.59,
  vec3<f32>(.99,.99,1.0),2u,&hit);
 sphere(ray,vec3<f32>(.20,-1.08,-1.89),.27,
  vec3<f32>(.92,.62,.30),0u,&hit);
 // The area emitter is slightly below the ceiling.
 if(abs(ray.direction.y)>0.00000001){
  let t=(1.365-ray.origin.y)/ray.direction.y;
  if(t>E&&t<hit.t){
   let pos=ray.origin+ray.direction*t;
   if(pos.x>=-.45&&pos.x<=.45&&pos.z>=-3.35&&pos.z<=-2.35){
    let front=ray.direction.y>0.0;
    hit.t=t;hit.position=pos;
    hit.normal=vec3<f32>(0.0,-1.0,0.0);
    hit.color=vec3<f32>(1.0);hit.material=3u;
    hit.emission=select(vec3<f32>(0.0),
      vec3<f32>(18.0,16.0,13.0)*params.lightStrength,front);
    hit.frontFace=front;
   }
  }
 }
 wall(ray,0u,-1.72,vec3<f32>(1.0,0.0,0.0),vec3<f32>(.80,.12,.13),&hit);
 wall(ray,0u,1.72,vec3<f32>(-1.0,0.0,0.0),vec3<f32>(.12,.70,.25),&hit);
 wall(ray,1u,-1.37,vec3<f32>(0.0,1.0,0.0),vec3<f32>(.84,.82,.80),&hit);
 wall(ray,1u,1.37,vec3<f32>(0.0,-1.0,0.0),vec3<f32>(.85,.85,.83),&hit);
 wall(ray,2u,-4.66,vec3<f32>(0.0,0.0,1.0),vec3<f32>(.79,.81,.87),&hit);
 return hit;
}
fn nextDirection(normal:vec3<f32>,seed:ptr<function,u32>)->vec3<f32>{
 let r=sqrt(rng(seed)),phi=2.0*PI*rng(seed);
 let t=normalize(cross(select(vec3<f32>(1.0,0.0,0.0),
   vec3<f32>(0.0,1.0,0.0),abs(normal.y)<.9),normal));
 let b=cross(normal,t);
 return normalize(t*r*cos(phi)+b*r*sin(phi)+normal*sqrt(max(0.0,1.0-r*r)));
}
fn nextEvent(hit:Hit,seed:ptr<function,u32>)->vec3<f32>{
 let sampled=vec3<f32>((rng(seed)-.5)*.9,1.365,-3.35+rng(seed));
 let delta=sampled-hit.position;
 let squared=dot(delta,delta);
 if(squared<0.0000001){return vec3<f32>(0.0);}
 let distance=sqrt(squared),wi=delta/distance;
 let cosN=max(0.0,dot(hit.normal,wi));
 let cosL=max(0.0,wi.y);
 if(cosN<=0.0||cosL<=0.0){return vec3<f32>(0.0);}
 let shadow=Ray(hit.position+hit.normal*(E*2.0),wi);
 if(traceScene(shadow,max(E,distance-E*4.0)).material!=9u){
  return vec3<f32>(0.0);
 }
 let estimate=cosN*cosL*.9/(PI*squared);
 return hit.color*vec3<f32>(18.0,16.0,13.0)*
   (params.lightStrength*estimate);
}
fn fresnel(c:f32,eta:f32)->f32{
 let a=(1.0-eta)/(1.0+eta);
 let r=a*a;
 return r+(1.0-r)*pow(1.0-clamp(c,0.0,1.0),5.0);
}
`+emitInlineCircuitWgsl(circuit.circuit,circuit.outputWires,'handwiredMaterialTransport')+`
fn travel(rayInitial:Ray,seed:ptr<function,u32>)->vec3<f32>{
 var ray=rayInitial;
 var throughput=vec3<f32>(1.0);
 var radiance=vec3<f32>(0.0);
 var previousSpecular=true;
 for(var bounce=0u;bounce<params.maxBounces;bounce=bounce+1u){
  let hit=traceScene(ray,1.0e20);
  if(hit.material==9u){break;}
  if(hit.material==3u){
   if(previousSpecular){radiance=radiance+throughput*hit.emission;}
   break;
  }
  if(hit.material==0u){
    radiance=radiance+throughput*nextEvent(hit,seed);
  }
  var transportSource:array<f32,6>;
  transportSource[0u]=throughput.x;transportSource[1u]=throughput.y;
  transportSource[2u]=throughput.z;
  transportSource[3u]=hit.color.x;transportSource[4u]=hit.color.y;
  transportSource[5u]=hit.color.z;
  let attenuated=handwiredMaterialTransport(transportSource);
  throughput=vec3<f32>(attenuated[0u],attenuated[1u],attenuated[2u]);
  if(hit.material==0u){
   ray=Ray(hit.position+hit.normal*(E*3.0),nextDirection(hit.normal,seed));
   previousSpecular=false;
  }else if(hit.material==1u){
   let dir=reflect(ray.direction,hit.normal);
   ray=Ray(hit.position+dir*(E*3.0),dir);
   previousSpecular=true;
  }else{
   let eta=select(params.ior,1.0/params.ior,hit.frontFace);
   let cosTheta=min(1.0,-dot(ray.direction,hit.normal));
   let t=refract(ray.direction,hit.normal,eta);
   let totalReflection=dot(t,t)<.01;
   let direction=select(t,reflect(ray.direction,hit.normal),
      totalReflection||rng(seed)<fresnel(cosTheta,eta));
   direction=normalize(direction);
   ray=Ray(hit.position+direction*(E*3.0),direction);
   previousSpecular=true;
  }
  if(bounce>=3u){
   let p=clamp(max(throughput.x,max(throughput.y,throughput.z)),.05,.98);
   if(rng(seed)>p){break;}
   throughput=throughput/p;
  }
 }
 return radiance;
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 if(id.x>=params.width||id.y>=params.height){return;}
 var seed=(id.x+id.y*params.width+1u)*747796405u+
   (params.sampleIndex+1u)*2891336453u;
 let jitter=vec2<f32>(rng(&seed),rng(&seed));
 let uv=(vec2<f32>(f32(id.x),f32(id.y))+jitter)/
  vec2<f32>(f32(params.width),f32(params.height));
 let x=(uv.x*2.0-1.0)*(f32(params.width)/f32(params.height))*tan(PI*48.0/360.0);
 let y=(1.0-2.0*uv.y)*tan(PI*48.0/360.0);
 let cy=cos(params.yaw);let sy=sin(params.yaw);
 let cp=cos(params.pitch);let sp=sin(params.pitch);
 let raw=vec3<f32>(x,y,-1.0);
 let rotated=vec3<f32>(cy*raw.x+sy*raw.z,
   cp*raw.y-sp*(cy*raw.z-sy*raw.x),
   sp*raw.y+cp*(cy*raw.z-sy*raw.x));
 let value=travel(Ray(vec3<f32>(0.0,0.0,2.4),normalize(rotated)),&seed);
 let index=id.y*params.width+id.x;
 accumulation[index]=accumulation[index]+vec4<f32>(value,1.0);
}
`

export const lightDisplayWgsl=`
struct Params {
 width:u32,height:u32,sampleIndex:u32,maxBounces:u32,
 yaw:f32,pitch:f32,exposure:f32,lightStrength:f32,
 ior:f32,pad0:f32,pad1:u32,pad2:u32,
};
@group(0) @binding(0) var<storage,read> accumulation:array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params:Params;
struct VertexOut{
 @builtin(position) position:vec4<f32>,
 @location(0) uv:vec2<f32>,
};
@vertex fn vs(@builtin(vertex_index) i:u32)->VertexOut{
 let points=array<vec2<f32>,6>(vec2<f32>(-1.0,-1.0),
   vec2<f32>(1.0,-1.0),vec2<f32>(-1.0,1.0),
   vec2<f32>(-1.0,1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0));
 var out:VertexOut;
 let p=points[i];
 out.position=vec4<f32>(p,0.0,1.0);
 out.uv=vec2<f32>((p.x+1.0)*.5,(1.0-p.y)*.5);
 return out;
}
@fragment fn fs(input:VertexOut)->@location(0) vec4<f32>{
 let x=min(u32(input.uv.x*f32(params.width)),params.width-1u);
 let y=min(u32(input.uv.y*f32(params.height)),params.height-1u);
 let sum=accumulation[y*params.width+x];
 let linear=sum.xyz/max(sum.w,1.0);
 let mapped=vec3<f32>(1.0)-exp(-linear*params.exposure);
 return vec4<f32>(pow(clamp(mapped,vec3<f32>(0.0),
   vec3<f32>(1.0)),vec3<f32>(1.0/2.2)),1.0);
}
`
export type LightSettings={exposure:number,lightStrength:number,
  ior:number,maxBounces:number,yaw:number,pitch:number}
export const lightDefaults:LightSettings={exposure:1.3,lightStrength:1.0,
  ior:1.5,maxBounces:5,yaw:0,pitch:0}
export class LightGpuRenderer {
 readonly gpu:GpuAllocation
 readonly width:number
 readonly height:number
 readonly canvas:HTMLCanvasElement
 readonly adapterName:string
 readonly accumulationBuffer:GPUBuffer
 readonly settings:LightSettings
 private readonly paramsBuffer:GPUBuffer
 private readonly computePipeline:GPUComputePipeline
 private readonly renderPipeline:GPURenderPipeline
 private readonly computeBind:GPUBindGroup
 private readonly renderBind:GPUBindGroup
 private readonly context:GPUCanvasContext
 private sampleIndex=0
 private disposed=false
 private constructor(gpu:GpuAllocation,canvas:HTMLCanvasElement,width:number,height:number,
   settings:LightSettings,accumulation:GPUBuffer,uniform:GPUBuffer,
   compute:GPUComputePipeline,render:GPURenderPipeline,
   computeBind:GPUBindGroup,renderBind:GPUBindGroup,context:GPUCanvasContext){
   this.gpu=gpu;this.canvas=canvas;this.width=width;this.height=height
   this.adapterName=gpu.adapter.name;this.settings={...settings}
   this.accumulationBuffer=accumulation;this.paramsBuffer=uniform
   this.computePipeline=compute;this.renderPipeline=render
   this.computeBind=computeBind;this.renderBind=renderBind;this.context=context
 }
 static async create(canvas:HTMLCanvasElement,width:number,height:number,
   settings:LightSettings=lightDefaults):Promise<LightGpuRenderer>{
   if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||
     width<48||height<48||width*height>1048576)
     throw new Error('invalid path tracing resolution')
   const gpu=await requestStrictGpu('Handwired Neural Light')
   try{
     if(Math.ceil(width/8)>gpu.device.limits.maxComputeWorkgroupsPerDimension||
       Math.ceil(height/8)>gpu.device.limits.maxComputeWorkgroupsPerDimension)
       throw new Error('path tracing workgroups exceed adapter limits')
     const bits={STORAGE:128,COPY_DST:8,COPY_SRC:4,UNIFORM:64}
     const accumulation=gpu.allocate(width*height*16,
       bits.STORAGE|bits.COPY_DST|bits.COPY_SRC,'Progressive photon radiance')
     const uniform=gpu.allocate(48,bits.UNIFORM|bits.COPY_DST,'Path tracing parameters')
     const computeShader=gpu.device.createShaderModule({code:lightKernelWgsl,label:'Handwired path tracing bounce transport'})
     const displayShader=gpu.device.createShaderModule({code:lightDisplayWgsl,label:'Path tracer radiance tonemap'})
     for(const module of [computeShader,displayShader]){
       const errors=(await module.getCompilationInfo()).messages.filter(x=>x.type==='error')
       if(errors.length)throw new Error('Path tracer WGSL: '+errors.map(x=>x.message).join('; '))
     }
     const compute=await gpu.device.createComputePipelineAsync({
       layout:'auto',compute:{module:computeShader,entryPoint:'main'}})
     const context=canvas.getContext('webgpu') as GPUCanvasContext|null
     if(context===null)throw new Error('No WebGPU canvas context')
     const format=navigator.gpu.getPreferredCanvasFormat()
     context.configure({device:gpu.device,format,alphaMode:'opaque'})
     const render=await gpu.device.createRenderPipelineAsync({
       layout:'auto',vertex:{module:displayShader,entryPoint:'vs'},
       fragment:{module:displayShader,entryPoint:'fs',targets:[{format}]},
       primitive:{topology:'triangle-list'},
     })
     const bind=(layout:GPUBindGroupLayout)=>gpu.device.createBindGroup({
       layout,entries:[{binding:0,resource:{buffer:accumulation}},
         {binding:1,resource:{buffer:uniform}}]})
     const out=new LightGpuRenderer(gpu,canvas,width,height,settings,accumulation,
       uniform,compute,render,bind(compute.getBindGroupLayout(0)),
       bind(render.getBindGroupLayout(0)),context)
     out.reset()
     return out
   }catch(error){gpu.dispose();throw error}
 }
 private check(){if(this.disposed)throw new Error('path tracer disposed');this.gpu.check()}
 get samplesPerPixel(){return this.sampleIndex}
 get gpuBytes(){return this.gpu.bytesAllocated()}
 update(next:Partial<LightSettings>){
  this.check()
  const settings={...this.settings,...next}
  if(![settings.exposure,settings.lightStrength,settings.ior,settings.yaw,
    settings.pitch].every(Number.isFinite)||
    settings.exposure<.01||settings.exposure>12||
    settings.lightStrength<0||settings.lightStrength>8||
    settings.ior<1||settings.ior>2.5||
    Math.abs(settings.yaw)>3||Math.abs(settings.pitch)>1.2||
    !Number.isInteger(settings.maxBounces)||
    settings.maxBounces<1||settings.maxBounces>10)throw new Error('invalid path tracing controls')
  Object.assign(this.settings,settings)
  this.reset()
 }
 reset(){
  this.check()
  this.sampleIndex=0
  const encoder=this.gpu.device.createCommandEncoder({label:'Reset progressive light accumulator'})
  encoder.clearBuffer(this.accumulationBuffer)
  this.gpu.device.queue.submit([encoder.finish()])
 }
 private writeParams(){
  const data=new ArrayBuffer(48),view=new DataView(data)
  view.setUint32(0,this.width,true);view.setUint32(4,this.height,true)
  view.setUint32(8,this.sampleIndex,true)
  view.setUint32(12,this.settings.maxBounces,true)
  for(const [index,value] of [this.settings.yaw,this.settings.pitch,
    this.settings.exposure,this.settings.lightStrength,this.settings.ior].entries())
    view.setFloat32(16+index*4,value,true)
  this.gpu.device.queue.writeBuffer(this.paramsBuffer,0,data)
 }
 frame(){
  this.check()
  if(this.sampleIndex>=1_000_000)throw new Error('sample counter limit reached')
  this.writeParams()
  this.canvas.width=this.width;this.canvas.height=this.height
  const encoder=this.gpu.device.createCommandEncoder({label:'Handwired progressive path sample'})
  const compute=encoder.beginComputePass()
  compute.setPipeline(this.computePipeline);compute.setBindGroup(0,this.computeBind)
  compute.dispatchWorkgroups(Math.ceil(this.width/8),Math.ceil(this.height/8))
  compute.end()
  const pass=encoder.beginRenderPass({colorAttachments:[{
    view:this.context.getCurrentTexture().createView(),
    loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:1},
  }]})
  pass.setPipeline(this.renderPipeline)
  pass.setBindGroup(0,this.renderBind);pass.draw(6);pass.end()
  this.gpu.device.queue.submit([encoder.finish()])
  this.sampleIndex++
 }
 async flush(){this.check();await this.gpu.device.queue.onSubmittedWorkDone();this.check()}
 /** Sparse inspection, never in the regular rendering path. */
 async readAccumulation():Promise<Float32Array>{
  this.check()
  const bytes=this.width*this.height*16
  const stage=this.gpu.device.createBuffer({size:bytes,usage:8|1})
  try{
    const encoder=this.gpu.device.createCommandEncoder()
    encoder.copyBufferToBuffer(this.accumulationBuffer,0,stage,0,bytes)
    this.gpu.device.queue.submit([encoder.finish()])
    await stage.mapAsync(1)
    const out=new Float32Array(stage.getMappedRange().slice(0))
    stage.unmap();this.check();return out
  }finally{stage.destroy()}
 }
 dispose(){
  if(this.disposed)return
  this.disposed=true
  this.context.unconfigure()
  this.gpu.dispose()
 }
}

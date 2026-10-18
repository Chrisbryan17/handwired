import {BondedStructureGpu} from './fracture-gpu.ts'

/** Real GPU-resident structural visualization. Every rendered beam reads
 * its two node positions and irreversible bond state from live GPU storage.
 * GPU-fractured bonds are excluded by clip-space culling, not a timeline.
 */
export const fractureVisualWgsl=`
struct Camera {
 right:vec4<f32>,up:vec4<f32>,forward:vec4<f32>,eye:vec4<f32>,projection:vec4<f32>,
};
@group(0) @binding(0) var<storage,read> nodes:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> endpoints:array<vec2<u32>>;
@group(0) @binding(2) var<storage,read> fracture:array<vec4<f32>>;
@group(0) @binding(3) var<uniform> camera:Camera;
struct VertexOut{
 @builtin(position) position:vec4<f32>,
 @location(0) normal:vec3<f32>,
 @location(1) damage:f32,
 @location(2) location:vec3<f32>,
};
const PI:f32=3.141592653589793;
const SIDES:u32=10u;
@vertex fn vs(@builtin(vertex_index) vertex:u32,
 @builtin(instance_index) instance:u32)->VertexOut{
 var out:VertexOut;
 let bond=fracture[instance];
 if(bond.w<=0.0){
  out.position=vec4<f32>(0.0,0.0,-2.0,1.0);
  out.normal=vec3<f32>(0.0,1.0,0.0);
  out.damage=1.0;out.location=vec3<f32>(0.0);
  return out;
 }
 let pair=endpoints[instance];
 let a=nodes[pair.x].xyz,b=nodes[pair.y].xyz;
 let axis=b-a;
 let direction=normalize(axis);
 let tangent=normalize(cross(direction,select(vec3<f32>(1.0,0.0,0.0),
   vec3<f32>(0.0,1.0,0.0),abs(direction.y)<.90)));
 let bitangent=cross(direction,tangent);
 let side=vertex/6u,corner=vertex%6u;
 let u=side,up=side+1u;
 var end=0.0;
 if(corner==1u||corner==3u||corner==4u){end=1.0;}
 if(corner==2u||corner==4u||corner==5u){u=up;}
 let angle=2.0*PI*f32(u)/f32(SIDES);
 let radial=tangent*cos(angle)+bitangent*sin(angle);
 let world=mix(a,b,end)+radial*.055;
 let relative=world-camera.eye.xyz;
 let depth=dot(relative,camera.forward.xyz);
 out.position=vec4<f32>(
  dot(relative,camera.right.xyz)*camera.projection.x,
  dot(relative,camera.up.xyz)*camera.projection.y,
  depth*camera.projection.z-camera.projection.w,depth);
 out.normal=radial;out.damage=clamp(bond.z,0.0,1.0);out.location=world;
 return out;
}
@fragment fn fs(in:VertexOut)->@location(0) vec4<f32>{
 let n=normalize(in.normal);
 let light=normalize(vec3<f32>(-.36,.84,.72));
 let lambert=max(.08,dot(n,light));
 let hot=clamp(in.damage*2.1,0.0,1.0);
 let material=mix(vec3<f32>(.20,.76,.95),vec3<f32>(1.0,.28,.13),hot);
 let view=normalize(camera.eye.xyz-in.location);
 let rim=pow(1.0-max(0.0,dot(n,view)),2.5);
 let rgb=material*(.25+.67*lambert)+vec3<f32>(.55,.84,1.0)*rim*.20;
 return vec4<f32>(pow(clamp(rgb,vec3<f32>(0.0),vec3<f32>(1.0)),vec3<f32>(1.0/2.2)),1.0);
}
`
const unit=(a:readonly number[])=>{
 const size=Math.hypot(...a)
 if(size<1e-8)throw new Error('invalid fracture camera')
 return a.map(v=>v/size)
}
const cross=(a:readonly number[],b:readonly number[])=>[
 a[1]!*b[2]!-a[2]!*b[1]!,
 a[2]!*b[0]!-a[0]!*b[2]!,
 a[0]!*b[1]!-a[1]!*b[0]!,
]
export class FractureRenderer{
 readonly solver:BondedStructureGpu
 readonly canvas:HTMLCanvasElement
 yaw=.25
 pitch=.38
 distance=13
 target:[number,number,number]=[0,2.4,0]
 private readonly context:GPUCanvasContext
 private readonly camera:GPUBuffer
 private readonly pipeline:GPURenderPipeline
 private readonly bind:GPUBindGroup
 private depth:GPUTexture|null=null
 private disposed=false
 private constructor(solver:BondedStructureGpu,canvas:HTMLCanvasElement,
  context:GPUCanvasContext,camera:GPUBuffer,pipeline:GPURenderPipeline,
  bind:GPUBindGroup){
  this.solver=solver;this.canvas=canvas;this.context=context
  this.camera=camera;this.pipeline=pipeline;this.bind=bind
 }
 static async create(solver:BondedStructureGpu,canvas:HTMLCanvasElement){
  const device=solver.gpu.device
  const context=canvas.getContext('webgpu') as GPUCanvasContext|null
  if(context===null)throw new Error('GPU fracture canvas unavailable')
  context.configure({device,format:navigator.gpu.getPreferredCanvasFormat(),alphaMode:'opaque'})
  try{
   const camera=solver.gpu.allocate(80,64|8,'Fracture beam camera')
   const shader=device.createShaderModule({code:fractureVisualWgsl})
   const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
   if(errors.length)throw new Error('Fracture display shader: '+errors.map(e=>e.message).join('; '))
   const pipeline=await device.createRenderPipelineAsync({
    layout:'auto',vertex:{module:shader,entryPoint:'vs'},
    fragment:{module:shader,entryPoint:'fs',
      targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},
    primitive:{topology:'triangle-list',cullMode:'none'},
    depthStencil:{format:'depth24plus',depthWriteEnabled:true,depthCompare:'less'},
   })
   const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:solver.nodePositions}},
    {binding:1,resource:{buffer:solver.bondEndpoints}},
    {binding:2,resource:{buffer:solver.fractureState}},
    {binding:3,resource:{buffer:camera}},
   ]})
   return new FractureRenderer(solver,canvas,context,camera,pipeline,bind)
  }catch(e){context.unconfigure();throw e}
 }
 private updateCamera(){
  const r=this.canvas.getBoundingClientRect()
  const w=Math.max(240,Math.min(1300,Math.round(r.width)))
  const h=Math.max(220,Math.min(760,Math.round(r.height)))
  if(w!==this.canvas.width||h!==this.canvas.height){
   this.canvas.width=w;this.canvas.height=h
   this.depth?.destroy()
   this.depth=this.solver.gpu.device.createTexture({
    size:[w,h],usage:16,format:'depth24plus'})
  }
  const eye=[
   this.target[0]+Math.sin(this.yaw)*Math.cos(this.pitch)*this.distance,
   this.target[1]+Math.sin(this.pitch)*this.distance,
   this.target[2]+Math.cos(this.yaw)*Math.cos(this.pitch)*this.distance,
  ]
  const forward=unit(this.target.map((v,i)=>v-eye[i]!))
  const right=unit(cross(forward,[0,1,0]))
  const up=unit(cross(right,forward))
  const f=1/Math.tan(Math.PI*51/360),near=.1,far=180
  this.solver.gpu.device.queue.writeBuffer(this.camera,0,Float32Array.of(
   ...right,0,...up,0,...forward,0,...eye,0,
   f*h/w,f,far/(far-near),near*far/(far-near)
  ))
 }
 render(){
  if(this.disposed)throw new Error('fracture display disposed')
  this.updateCamera()
  const encoder=this.solver.gpu.device.createCommandEncoder()
  const pass=encoder.beginRenderPass({
   colorAttachments:[{view:this.context.getCurrentTexture().createView(),
     loadOp:'clear',storeOp:'store',clearValue:{r:.012,g:.027,b:.059,a:1}}],
   depthStencilAttachment:{view:this.depth!.createView(),
     depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'},
  })
  pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bind)
  pass.draw(10*6,this.solver.bondCount)
  pass.end()
  this.solver.gpu.device.queue.submit([encoder.finish()])
 }
 dispose(){if(this.disposed)return;this.disposed=true;this.depth?.destroy();this.context.unconfigure()}
}

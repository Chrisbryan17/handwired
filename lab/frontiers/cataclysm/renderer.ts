import {CataclysmSphGpu} from './solver.ts'
import {cataclysmRenderWgsl} from './render-shader.ts'
const norm=(v:readonly number[])=>{
 const length=Math.hypot(...v)
 if(length<1e-6)throw new Error('invalid camera vector')
 return v.map(x=>x/length)
}
const cross=(a:readonly number[],b:readonly number[])=>[
 a[1]!*b[2]!-a[2]!*b[1]!,a[2]!*b[0]!-a[0]!*b[2]!,
 a[0]!*b[1]!-a[1]!*b[0]!
]
export class CataclysmRenderer{
 readonly engine:CataclysmSphGpu
 readonly canvas:HTMLCanvasElement
 yaw=.42
 pitch=.26
 distance=17
 target:[number,number,number]=[-1,1,0]
 private readonly context:GPUCanvasContext
 private readonly camera:GPUBuffer
 private readonly pipeline:GPURenderPipeline
 private readonly bind:GPUBindGroup
 private depth:GPUTexture|null=null
 private disposed=false
 private constructor(engine:CataclysmSphGpu,canvas:HTMLCanvasElement,
   context:GPUCanvasContext,camera:GPUBuffer,pipeline:GPURenderPipeline,
   bind:GPUBindGroup){
   this.engine=engine;this.canvas=canvas;this.context=context
   this.camera=camera;this.pipeline=pipeline;this.bind=bind
 }
 static async create(engine:CataclysmSphGpu,canvas:HTMLCanvasElement){
  const device=engine.gpu.device
  const context=canvas.getContext('webgpu') as GPUCanvasContext|null
  if(context===null)throw new Error('Water stage needs real WebGPU canvas')
  const format=navigator.gpu.getPreferredCanvasFormat()
  context.configure({device,format,alphaMode:'opaque'})
  try{
   const camera=engine.gpu.allocate(80,64|8,'3D water orbital camera')
   const shader=device.createShaderModule({code:cataclysmRenderWgsl,label:'Real 3D WCSPH water particle render'})
   const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
   if(errors.length)throw new Error('Fluid shader WGSL: '+errors.map(x=>x.message).join('; '))
   const pipeline=await device.createRenderPipelineAsync({
    layout:'auto',vertex:{module:shader,entryPoint:'vs'},
    fragment:{module:shader,entryPoint:'fs',targets:[{format}]},
    primitive:{topology:'triangle-list',cullMode:'none'},
    depthStencil:{format:'depth24plus',depthWriteEnabled:true,depthCompare:'less'},
   })
   const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:engine.grid.positions}},
    {binding:1,resource:{buffer:engine.velocityBuffer}},
    {binding:2,resource:{buffer:camera}},
   ]})
   return new CataclysmRenderer(engine,canvas,context,camera,pipeline,bind)
  }catch(e){context.unconfigure();throw e}
 }
 private updateCamera(){
  const rect=this.canvas.getBoundingClientRect()
  const w=Math.max(260,Math.min(1300,Math.round(rect.width)))
  const h=Math.max(230,Math.min(780,Math.round(rect.height)))
  if(w!==this.canvas.width||h!==this.canvas.height){
   this.canvas.width=w;this.canvas.height=h
   this.depth?.destroy()
   this.depth=this.engine.gpu.device.createTexture({
    size:[w,h],format:'depth24plus',usage:16,
    label:'Water particle depth'})
  }
  const eye=[
   this.target[0]+Math.sin(this.yaw)*Math.cos(this.pitch)*this.distance,
   this.target[1]+Math.sin(this.pitch)*this.distance,
   this.target[2]+Math.cos(this.yaw)*Math.cos(this.pitch)*this.distance
  ]
  const forward=norm(this.target.map((x,i)=>x-eye[i]!))
  const right=norm(cross(forward,[0,1,0]))
  const up=norm(cross(right,forward))
  const f=1/Math.tan(Math.PI*52/360),near=.1,far=180
  this.engine.gpu.device.queue.writeBuffer(this.camera,0,Float32Array.of(
   ...right,0,...up,0,...forward,0,...eye,0,
   f*h/w,f,far/(far-near),near*far/(far-near)
  ))
 }
 render(){
  if(this.disposed)throw new Error('fluid viewer disposed')
  this.updateCamera()
  const encoder=this.engine.gpu.device.createCommandEncoder()
  const pass=encoder.beginRenderPass({
   colorAttachments:[{view:this.context.getCurrentTexture().createView(),
    loadOp:'clear',storeOp:'store',clearValue:{r:.018,g:.038,b:.074,a:1}}],
   depthStencilAttachment:{view:this.depth!.createView(),
    depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}
  })
  pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bind)
  pass.draw(6,this.engine.count)
  pass.end()
  this.engine.gpu.device.queue.submit([encoder.finish()])
 }
 dispose(){
  if(this.disposed)return
  this.disposed=true;this.depth?.destroy();this.context.unconfigure()
 }
}

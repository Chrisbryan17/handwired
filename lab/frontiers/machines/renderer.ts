import {BatchedLegsGpu} from './gpu.ts'
import {legPopulationRenderWgsl} from './render-shader.ts'
const unit=(v:readonly number[])=>{
 const length=Math.hypot(...v)
 if(length<1e-6)throw new Error('degenerate robotics camera')
 return v.map(x=>x/length)
}
const cross=(a:readonly number[],b:readonly number[])=>[
 a[1]!*b[2]!-a[2]!*b[1]!,
 a[2]!*b[0]!-a[0]!*b[2]!,
 a[0]!*b[1]!-a[1]!*b[0]!
]
export class MachinesRenderer {
 readonly engine:BatchedLegsGpu
 readonly canvas:HTMLCanvasElement
 yaw=.46
 pitch=.42
 distance=32
 target:[number,number,number]=[0,1,0]
 private readonly context:GPUCanvasContext
 private readonly camera:GPUBuffer
 private readonly binding:readonly [GPUBindGroup,GPUBindGroup]
 private readonly ground:GPURenderPipeline
 private readonly rods:GPURenderPipeline
 private readonly joints:GPURenderPipeline
 private depth:GPUTexture|null=null
 private disposed=false
 private constructor(engine:BatchedLegsGpu,canvas:HTMLCanvasElement,
  context:GPUCanvasContext,camera:GPUBuffer,
  binding:readonly [GPUBindGroup,GPUBindGroup],
  ground:GPURenderPipeline,rods:GPURenderPipeline,joints:GPURenderPipeline){
   this.engine=engine;this.canvas=canvas;this.context=context
   this.camera=camera;this.binding=binding
   this.ground=ground;this.rods=rods;this.joints=joints
 }
 static async create(engine:BatchedLegsGpu,canvas:HTMLCanvasElement){
  const device=engine.gpu.device
  const context=canvas.getContext('webgpu') as GPUCanvasContext|null
  if(context===null)throw new Error('Real WebGPU robotics canvas is unavailable')
  const format=navigator.gpu.getPreferredCanvasFormat()
  context.configure({device,format,alphaMode:'opaque'})
  try{
   const camera=engine.gpu.allocate(112,64|8,'Neural Machines orbital camera')
   const shader=device.createShaderModule({code:legPopulationRenderWgsl,
    label:'Instanced physical 3D articulated population geometry'})
   const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
   if(errors.length)throw new Error('Articulated scene WGSL: '+errors.map(x=>x.message).join('; '))
   const pipeline=async(entryPoint:string)=>device.createRenderPipelineAsync({
    layout:'auto',vertex:{module:shader,entryPoint},
    fragment:{module:shader,entryPoint:'fragment',targets:[{format}]},
    primitive:{topology:'triangle-list',cullMode:'none'},
    depthStencil:{format:'depth24plus',depthWriteEnabled:true,depthCompare:'less'},
   })
   const [ground,rods,joints]=await Promise.all([
    pipeline('ground'),pipeline('rod'),pipeline('joint')])
   const binding=engine.renderJointBuffers.map(b=>device.createBindGroup({
    layout:ground.getBindGroupLayout(0),entries:[
     {binding:0,resource:{buffer:b}},
     {binding:1,resource:{buffer:camera}},
    ]})) as [GPUBindGroup,GPUBindGroup]
   const viewer=new MachinesRenderer(engine,canvas,context,camera,binding,
    ground,rods,joints)
   viewer.distance=Math.max(25,Math.min(250,Math.sqrt(engine.count)*3.5))
   return viewer
  }catch(error){context.unconfigure();throw error}
 }
 private updateCamera(){
  const rect=this.canvas.getBoundingClientRect()
  const width=Math.max(256,Math.min(1500,Math.round(rect.width)))
  const height=Math.max(240,Math.min(850,Math.round(rect.height)))
  if(this.canvas.width!==width||this.canvas.height!==height){
   this.canvas.width=width;this.canvas.height=height
   this.depth?.destroy()
   this.depth=this.engine.gpu.device.createTexture({
    size:[width,height],usage:16,format:'depth24plus',
    label:'Neural Machines physical renderer depth'})
  }
  const eye=[
   this.target[0]+Math.sin(this.yaw)*Math.cos(this.pitch)*this.distance,
   this.target[1]+Math.sin(this.pitch)*this.distance,
   this.target[2]+Math.cos(this.yaw)*Math.cos(this.pitch)*this.distance,
  ]
  const forward=unit(this.target.map((x,i)=>x-eye[i]!))
  const right=unit(cross(forward,[0,1,0]))
  const up=unit(cross(right,forward))
  const f=1/Math.tan(Math.PI*52/360),near=.1,far=900
  const columns=Math.ceil(Math.sqrt(this.engine.count))
  const rows=Math.ceil(this.engine.count/columns)
  const p=this.engine.parameters
  this.engine.gpu.device.queue.writeBuffer(this.camera,0,Float32Array.of(
   ...right,0,...up,0,...forward,0,...eye,0,
   f*height/width,f,far/(far-near),near*far/(far-near),
   columns,rows,2.25,p.groundHeight,
   p.length1,p.length2,p.hipHeight,.105
  ))
 }
 render(){
  if(this.disposed)throw new Error('robot display disposed')
  this.updateCamera()
  const encoder=this.engine.gpu.device.createCommandEncoder({
   label:'Physical linked-robot GPU geometry'})
  const pass=encoder.beginRenderPass({
   colorAttachments:[{view:this.context.getCurrentTexture().createView(),
    loadOp:'clear',storeOp:'store',clearValue:{r:.016,g:.023,b:.055,a:1}}],
   depthStencilAttachment:{view:this.depth!.createView(),
    depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'},
  })
  pass.setBindGroup(0,this.binding[this.engine.activeIndex]!)
  pass.setPipeline(this.ground);pass.draw(6)
  pass.setPipeline(this.rods);pass.draw(48,this.engine.count*2)
  pass.setPipeline(this.joints);pass.draw(216,this.engine.count*3)
  pass.end()
  this.engine.gpu.device.queue.submit([encoder.finish()])
 }
 dispose(){
  if(this.disposed)return
  this.disposed=true
  this.depth?.destroy()
  this.context.unconfigure()
 }
}

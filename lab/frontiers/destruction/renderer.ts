import {SphereContactGpu} from '../../../src/runtime/frontiers-contacts-gpu.ts'
import {destructionWgsl} from './shader.ts'
import type {SphereBodies} from '../../../src/runtime/frontiers-contacts.ts'

const verticesPerSphere=12*20*6
const length=(v:readonly number[])=>Math.hypot(...v)
const normalized=(v:readonly number[])=>{
 const norm=length(v)
 if(norm<1e-6)throw new Error('degenerate camera basis')
 return v.map(x=>x/norm)
}
const cross=(a:readonly number[],b:readonly number[])=>[
 a[1]!*b[2]!-a[2]!*b[1]!,
 a[2]!*b[0]!-a[0]!*b[2]!,
 a[0]!*b[1]!-a[1]!*b[0]!,
]
export class ContactSceneRenderer{
 readonly solver:SphereContactGpu
 readonly canvas:HTMLCanvasElement
 private readonly context:GPUCanvasContext
 private readonly camera:GPUBuffer
 private readonly pipeline:GPURenderPipeline
 private readonly bind:GPUBindGroup
 private depth:GPUTexture|null=null
 yaw=.35
 pitch=.27
 distance=20
 target:[number,number,number]=[0,0,0]
 private disposed=false
 private constructor(solver:SphereContactGpu,canvas:HTMLCanvasElement,
   context:GPUCanvasContext,camera:GPUBuffer,
   pipeline:GPURenderPipeline,bind:GPUBindGroup){
   this.solver=solver;this.canvas=canvas;this.context=context
   this.camera=camera;this.pipeline=pipeline;this.bind=bind
 }
 static async create(solver:SphereContactGpu,canvas:HTMLCanvasElement,
   initial:SphereBodies):Promise<ContactSceneRenderer>{
   const device=solver.grid.gpu.device
   const context=canvas.getContext('webgpu') as GPUCanvasContext|null
   if(context===null)throw new Error('missing WebGPU scene canvas context')
   const format=navigator.gpu.getPreferredCanvasFormat()
   context.configure({device,format,alphaMode:'opaque'})
   try{
     const color=solver.grid.gpu.allocate(solver.count*16,128|8,'Impact body surface colors')
     const camera=solver.grid.gpu.allocate(80,64|8,'Impact perspective camera')
     const values=new Float32Array(solver.count*4)
     for(let i=0;i<solver.count;i++){
       const inv=initial.velocities[i*4+3]!
       const palette=inv===0?[.18,.31,.39]:i%11===0?[1.0,.66,.25]:
         i%3===0?[.58,.43,1.0]:i%3===1?[.23,.9,.79]:[.44,.69,.98]
       values.set([...palette,1],i*4)
     }
     device.queue.writeBuffer(color,0,values)
     const shader=device.createShaderModule({code:destructionWgsl,label:'Lit GPU-resident contact spheres'})
     const errors=(await shader.getCompilationInfo()).messages.filter(e=>e.type==='error')
     if(errors.length)throw new Error('3D impact renderer WGSL: '+errors.map(x=>x.message).join('; '))
     const pipeline=await device.createRenderPipelineAsync({
       layout:'auto',vertex:{module:shader,entryPoint:'vs'},
       fragment:{module:shader,entryPoint:'fs',targets:[{format}]},
       primitive:{topology:'triangle-list',cullMode:'none'},
       depthStencil:{format:'depth24plus',depthWriteEnabled:true,depthCompare:'less'},
     })
     const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
       {binding:0,resource:{buffer:solver.grid.positions}},
       {binding:1,resource:{buffer:color}},
       {binding:2,resource:{buffer:camera}},
     ]})
     return new ContactSceneRenderer(solver,canvas,context,camera,pipeline,bind)
   }catch(error){context.unconfigure();throw error}
 }
 private updateCamera(){
   const rect=this.canvas.getBoundingClientRect()
   const width=Math.max(256,Math.min(1280,Math.round(rect.width)))
   const height=Math.max(210,Math.min(800,Math.round(rect.height)))
   if(this.canvas.width!==width||this.canvas.height!==height){
     this.canvas.width=width;this.canvas.height=height
     this.depth?.destroy()
     this.depth=this.solver.grid.gpu.device.createTexture({
       size:[width,height],format:'depth24plus',usage:16,label:'Contact render depth'
     })
   }
   const eye=[
     this.target[0]+Math.sin(this.yaw)*Math.cos(this.pitch)*this.distance,
     this.target[1]+Math.sin(this.pitch)*this.distance,
     this.target[2]+Math.cos(this.yaw)*Math.cos(this.pitch)*this.distance
   ]
   const forward=normalized(this.target.map((x,i)=>x-eye[i]!))
   const right=normalized(cross(forward,[0,1,0]))
   const up=normalized(cross(right,forward))
   const f=1/Math.tan(Math.PI*52/360),near=.1,far=250
   this.solver.grid.gpu.device.queue.writeBuffer(this.camera,0,Float32Array.of(
     ...right,0,...up,0,...forward,0,...eye,0,
     f*height/width,f,far/(far-near),near*far/(far-near)
   ))
 }
 render(){
   if(this.disposed)throw new Error('renderer disposed')
   this.updateCamera()
   const encoder=this.solver.grid.gpu.device.createCommandEncoder({label:'Contact laboratory lit 3D geometry'})
   const pass=encoder.beginRenderPass({
     colorAttachments:[{view:this.context.getCurrentTexture().createView(),
       loadOp:'clear',storeOp:'store',clearValue:{r:.025,g:.038,b:.075,a:1}}],
     depthStencilAttachment:{view:this.depth!.createView(),
       depthClearValue:1,depthLoadOp:'clear',depthStoreOp:'store'}
   })
   pass.setPipeline(this.pipeline)
   pass.setBindGroup(0,this.bind)
   pass.draw(verticesPerSphere,this.solver.count)
   pass.end()
   this.solver.grid.gpu.device.queue.submit([encoder.finish()])
 }
 dispose(){
   if(this.disposed)return
   this.disposed=true
   this.depth?.destroy()
   this.context.unconfigure()
 }
}

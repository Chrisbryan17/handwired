import type {UniverseGpuSolver} from './solver.ts'

const shader=`
struct Particle {position:vec4<f32>,velocity:vec4<f32>};
struct Camera {
  width:f32,height:f32,yaw:f32,pitch:f32,
  zoom:f32,scale:f32,count:u32,pad:u32,
};
@group(0) @binding(0) var<storage,read> bodies:array<Particle>;
@group(0) @binding(1) var<uniform> camera:Camera;
struct VertexOutput {
 @builtin(position) point:vec4<f32>,
 @location(0) uv:vec2<f32>,
 @location(1) color:vec3<f32>,
};
@vertex fn vertex(@builtin(vertex_index) vi:u32,
                  @builtin(instance_index) i:u32)->VertexOutput {
  let corners=array<vec2<f32>,6>(
    vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),vec2<f32>(-1.0,1.0),
    vec2<f32>(-1.0,1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0)
  );
  let particle=bodies[i];
  let c=cos(camera.yaw);let s=sin(camera.yaw);
  let cx=cos(camera.pitch);let sx=sin(camera.pitch);
  let x=c*particle.position.x+s*particle.position.z;
  let z=c*particle.position.z-s*particle.position.x;
  let y=cx*particle.position.y-sx*z;
  let rotatedZ=sx*particle.position.y+cx*z;
  let distance=max(.22,camera.zoom-rotatedZ);
  let aspect=camera.height/max(camera.width,1.0);
  let pointScale=select(.009,.032,particle.position.w>.10);
  let radius=camera.scale*pointScale/distance;
  let base=vec2<f32>(x*aspect,y)*1.6/distance;
  let corner=corners[vi];
  var output:VertexOutput;
  output.point=vec4<f32>(base+corner*radius*vec2<f32>(aspect,1.0),
    clamp(.5+rotatedZ*.03,0.0,1.0),1.0);
  output.uv=corner;
  let typeID=particle.velocity.w;
  let speed=min(length(particle.velocity.xyz),2.5);
  let blue=vec3<f32>(.17,.57,1.0);
  let rose=vec3<f32>(1.0,.36,.46);
  let golden=vec3<f32>(1.0,.93,.69);
  output.color=select(select(blue,rose,typeID>.5),golden,typeID>1.5)*
    (0.8+speed*.16);
  return output;
}
@fragment fn fragment(input:VertexOutput)->@location(0) vec4<f32> {
  let r=length(input.uv);
  if(r>1.0){discard;}
  let core=pow(max(0.0,1.0-r),1.5);
  let halo=.24*pow(max(0.0,1.0-r),3.5);
  return vec4<f32>(input.color*(core+halo)*1.25,1.0);
}
`
export class UniverseRenderer {
  readonly canvas:HTMLCanvasElement
  private readonly context:GPUCanvasContext
  private readonly device:GPUDevice
  private readonly pipeline:GPURenderPipeline
  private readonly cameraBuffer:GPUBuffer
  private readonly groups:readonly [GPUBindGroup,GPUBindGroup]
  private readonly solver:UniverseGpuSolver
  private disposed=false
  yaw=.2
  pitch=.15
  zoom=2.7
  scale=1.0
  private constructor(solver:UniverseGpuSolver,canvas:HTMLCanvasElement,
    context:GPUCanvasContext,pipeline:GPURenderPipeline,uniform:GPUBuffer,
    groups:[GPUBindGroup,GPUBindGroup]) {
    this.device=solver.gpu.device
    this.solver=solver;this.canvas=canvas;this.context=context
    this.pipeline=pipeline;this.cameraBuffer=uniform;this.groups=groups
  }
  static async create(solver:UniverseGpuSolver,canvas:HTMLCanvasElement):
    Promise<UniverseRenderer> {
    const device=solver.gpu.device
    const context=canvas.getContext('webgpu') as GPUCanvasContext | null
    if(!context)throw new Error('No canvas WebGPU context')
    const format=navigator.gpu.getPreferredCanvasFormat()
    context.configure({device,format,alphaMode:'opaque'})
    const module=device.createShaderModule({code:shader,label:'N-body star billboards'})
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error')
    if(errors.length)throw new Error('Star renderer WGSL: '+errors.map(m=>m.message).join('; '))
    const pipeline=await device.createRenderPipelineAsync({
      layout:'auto',
      vertex:{module,entryPoint:'vertex'},
      fragment:{module,entryPoint:'fragment',targets:[{format,blend:{
        color:{srcFactor:'one',dstFactor:'one',operation:'add'},
        alpha:{srcFactor:'one',dstFactor:'one',operation:'add'},
      }}]},
      primitive:{topology:'triangle-list'},
    })
    const uniform=solver.gpu.allocate(32,64|8,'Universe camera')
    const bind=(buffer:GPUBuffer)=>device.createBindGroup({
      layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer}},
        {binding:1,resource:{buffer:uniform}},
      ]})
    return new UniverseRenderer(solver,canvas,context,pipeline,uniform,
      [bind(solver.buffers[0]),bind(solver.buffers[1])])
  }
  render(advance=true) {
    if(this.disposed)throw new Error('Universe renderer disposed')
    this.solver.gpu.check()
    const pixelRatio=Math.min(2,window.devicePixelRatio||1)
    const width=Math.max(1,Math.floor(this.canvas.clientWidth*pixelRatio))
    const height=Math.max(1,Math.floor(this.canvas.clientHeight*pixelRatio))
    if(this.canvas.width!==width||this.canvas.height!==height){
      this.canvas.width=width;this.canvas.height=height
    }
    const data=new ArrayBuffer(32),view=new DataView(data)
    for(const [i,value] of [width,height,this.yaw,this.pitch,this.zoom,this.scale].entries())
      view.setFloat32(i*4,value,true)
    view.setUint32(24,this.solver.count,true)
    this.device.queue.writeBuffer(this.cameraBuffer,0,data)
    const encoder=this.device.createCommandEncoder({label:'Universe frame'})
    if(advance)this.solver.encodeStep(encoder)
    const pass=encoder.beginRenderPass({colorAttachments:[{
      view:this.context.getCurrentTexture().createView(),
      loadOp:'clear',storeOp:'store',
      clearValue:{r:.007,g:.011,b:.028,a:1},
    }]})
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0,this.groups[this.solver.currentIndex]!)
    pass.draw(6,this.solver.count)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }
  dispose(){
    if(this.disposed)return
    this.disposed=true
    this.context.unconfigure()
  }
}

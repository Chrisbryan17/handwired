import type {GenesisGpuSolver} from './solver.ts'

/** Direct GPU field visualization. No CPU image reconstruction or state readback.
 * Three nonlinear color ramps distinguish chemical concentrations and gradients.
 */
const shader = `
struct Params {width:f32,height:f32,mode:f32,padding:f32};
@group(0) @binding(0) var<storage,read> state:array<f32>;
@group(0) @binding(1) var<uniform> p:Params;
struct VertexOutput{
 @builtin(position) position:vec4<f32>,
 @location(0) uv:vec2<f32>,
};
@vertex fn vertex(@builtin(vertex_index) index:u32)->VertexOutput{
 let positions=array<vec2<f32>,6>(vec2<f32>(-1.0,-1.0),vec2<f32>(1.0,-1.0),
   vec2<f32>(-1.0,1.0),vec2<f32>(-1.0,1.0),vec2<f32>(1.0,-1.0),vec2<f32>(1.0,1.0));
 var output:VertexOutput;
 let pos=positions[index];
 output.position=vec4<f32>(pos,0.0,1.0);
 output.uv=vec2<f32>((pos.x+1.0)*.5,(1.0-pos.y)*.5);
 return output;
}
fn field(x:i32,y:i32,ch:u32)->f32{
 let ix=u32(clamp(x,0,i32(p.width)-1));
 let iy=u32(clamp(y,0,i32(p.height)-1));
 return state[(iy*u32(p.width)+ix)*2u+ch];
}
fn bilinear(pos:vec2<f32>,ch:u32)->f32{
 let x=i32(floor(pos.x));let y=i32(floor(pos.y));
 let tx=fract(pos.x);let ty=fract(pos.y);
 let top=mix(field(x,y,ch),field(x+1,y,ch),tx);
 let bottom=mix(field(x,y+1,ch),field(x+1,y+1,ch),tx);
 return mix(top,bottom,ty);
}
@fragment fn fragment(input:VertexOutput)->@location(0) vec4<f32>{
 let uv=input.uv*vec2<f32>(p.width,p.height)-vec2<f32>(.5,.5);
 let u=bilinear(uv,0u);let v=bilinear(uv,1u);
 let intensity=smoothstep(.09,.46,v);
 let high=smoothstep(.24,.75,v);
 let low=smoothstep(.05,.18,v);
 let deep=vec3<f32>(.015,.024,.060);
 var color=mix(deep,vec3<f32>(.12,.38,.66),low);
 color=mix(color,vec3<f32>(.22,.95,.82),intensity*.89);
 color=mix(color,vec3<f32>(1.0,.75,.38),high);
 if(p.mode>0.5 && p.mode<1.5) {
   color=mix(vec3<f32>(.06,.025,.11),vec3<f32>(.9,.16,.49),clamp(v*3.0,0.0,1.0));
   color+=vec3<f32>(.14,.48,.82)*clamp(1.0-u,0.0,1.0)*.8;
 }else if(p.mode>1.5){
   color=vec3<f32>(clamp(1.0-u,0.0,1.0),clamp(v*2.6,0.0,1.0),clamp(v*5.0,0.0,1.0));
 }
 return vec4<f32>(color,1.0);
}
`
export class GenesisRenderer {
  private readonly device:GPUDevice
  private readonly context:GPUCanvasContext
  private readonly pipeline:GPURenderPipeline
  private readonly groups:[GPUBindGroup,GPUBindGroup]
  private readonly uniform:GPUBuffer
  private disposed=false
  palette=0
  private constructor(private readonly solver:GenesisGpuSolver,
    private readonly canvas:HTMLCanvasElement,context:GPUCanvasContext,
    pipeline:GPURenderPipeline,uniform:GPUBuffer,groups:[GPUBindGroup,GPUBindGroup]){
    this.device=solver.gpu.device;this.context=context;this.pipeline=pipeline
    this.uniform=uniform;this.groups=groups
  }
  static async create(solver:GenesisGpuSolver,canvas:HTMLCanvasElement):Promise<GenesisRenderer>{
    const device=solver.gpu.device
    const context=canvas.getContext('webgpu') as GPUCanvasContext|null
    if(context===null)throw new Error('canvas WebGPU is unavailable')
    const format=navigator.gpu.getPreferredCanvasFormat()
    context.configure({device,format,alphaMode:'opaque'})
    const module=device.createShaderModule({code:shader,label:'Morphogen field renderer'})
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error')
    if(errors.length)throw new Error('Morphogen render WGSL: '+errors.map(m=>m.message).join('; '))
    const pipeline=await device.createRenderPipelineAsync({
      layout:'auto',vertex:{module,entryPoint:'vertex'},
      fragment:{module,entryPoint:'fragment',targets:[{format}]},
      primitive:{topology:'triangle-list'},
    })
    const uniform=solver.gpu.allocate(16,64|8,'Morphogen render parameters')
    const bind=(buffer:GPUBuffer)=>device.createBindGroup({
      layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer}},{binding:1,resource:{buffer:uniform}},
      ],
    })
    return new GenesisRenderer(solver,canvas,context,pipeline,uniform,
      [bind(solver.buffers[0]),bind(solver.buffers[1])])
  }
  render(advance=true){
    if(this.disposed)throw new Error('morphogen renderer disposed')
    this.solver.gpu.check()
    const dpr=Math.min(2,window.devicePixelRatio||1)
    const width=Math.max(1,Math.floor(this.canvas.clientWidth*dpr))
    const height=Math.max(1,Math.floor(this.canvas.clientHeight*dpr))
    if(this.canvas.width!==width||this.canvas.height!==height){
      this.canvas.width=width;this.canvas.height=height
    }
    const param=new Float32Array([this.solver.width,this.solver.height,this.palette,0])
    this.device.queue.writeBuffer(this.uniform,0,param)
    const encoder=this.device.createCommandEncoder({label:'Morphogen chemistry + display'})
    if(advance)this.solver.encodeStep(encoder)
    const pass=encoder.beginRenderPass({colorAttachments:[{
      view:this.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',
      clearValue:{r:.015,g:.02,b:.05,a:1},
    }]})
    pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.groups[this.solver.currentIndex]!)
    pass.draw(6);pass.end()
    this.device.queue.submit([encoder.finish()])
  }
  dispose(){if(this.disposed)return;this.disposed=true;this.context.unconfigure()}
}

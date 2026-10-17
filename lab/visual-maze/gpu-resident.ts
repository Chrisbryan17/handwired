import {buildMazeCircuit} from './model.ts'
import {emitInlineCircuitWgsl} from '../../src/runtime/webgpu-inline.ts'
import {beginSearch,type Maze} from './engine.ts'

const U={MAP_READ:1,COPY_SRC:4,COPY_DST:8,UNIFORM:64,STORAGE:128}
const M=1
const {circuit,outputWires}=buildMazeCircuit()
const inline=emitInlineCircuitWgsl(circuit,outputWires,'handwiredMaze')
const common=`
struct Settings{columns:u32,rows:u32,depth:u32,start:u32,goal:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<uniform> settings:Settings;
@group(0) @binding(1) var<storage,read> walls:array<u32>;
@group(0) @binding(2) var<storage,read> previous:array<vec4<f32>>;
fn frontier(x:i32,y:i32)->f32{
 if(x<0||y<0||x>=i32(settings.columns)||y>=i32(settings.rows)){return 0.0;}
 return previous[u32(y)*settings.columns+u32(x)].y;
}
`
const compute=common+
'@group(0) @binding(3) var<storage,read_write> next:array<vec4<f32>>;\n'+
inline+`
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=settings.columns*settings.rows){return;}
 let x=i32(i%settings.columns);let y=i32(i/settings.columns);
 let old=previous[i];
 var source:array<f32,6>;
 source[0u]=old.x;source[1u]=frontier(x,y-1);source[2u]=frontier(x+1,y);
 source[3u]=frontier(x,y+1);source[4u]=frontier(x-1,y);
 source[5u]=select(1.0,0.0,walls[i]!=0u);
 let result=handwiredMaze(source);
 let newFrontier=result[1u];
 let d=select(old.z,f32(settings.depth+1u),newFrontier>0.5);
 next[i]=vec4<f32>(result[0u],newFrontier,d,0.0);
}`
const render=`
struct Settings{columns:u32,rows:u32,depth:u32,start:u32,goal:u32,pad0:u32,pad1:u32,pad2:u32}
@group(0) @binding(0) var<uniform> settings:Settings;
@group(0) @binding(1) var<storage,read> walls:array<u32>;
@group(0) @binding(2) var<storage,read> state:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> pathMask:array<u32>;
struct VertexOut{@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>}
@vertex fn vs(@builtin(vertex_index) i:u32)->VertexOut {
 var positions=array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
 var out:VertexOut;
 out.position=vec4<f32>(positions[i],0.0,1.0);
 out.uv=vec2<f32>((positions[i].x+1.0)*0.5,(1.0-positions[i].y)*0.5);
 return out;
}
@fragment fn fs(@location(0) uv:vec2<f32>)->@location(0) vec4<f32>{
 let cx=clamp(uv.x,0.0,0.999999)*f32(settings.columns);
 let cy=clamp(uv.y,0.0,0.999999)*f32(settings.rows);
 let x=u32(cx);let y=u32(cy);let i=y*settings.columns+x;
 let frac=fract(vec2<f32>(cx,cy));
 var color=vec3<f32>(0.030,0.047,0.105);
 if(walls[i]==0u){
   color=vec3<f32>(0.082,0.11,0.22);
   let s=state[i];
   if(s.x>0.5){color=vec3<f32>(0.15,0.23,0.45);}
   if(s.y>0.5){color=vec3<f32>(0.59,0.96,0.82);}
   if(pathMask[i]==1u){color=vec3<f32>(0.91,0.79,0.42);}
   if(i==settings.start){color=vec3<f32>(1.0,0.96,0.79);}
   if(i==settings.goal){color=vec3<f32>(1.0,0.70,0.43);}
   if(min(min(frac.x,frac.y),min(1.0-frac.x,1.0-frac.y))<0.065){color=color*0.58;}
 }
 return vec4<f32>(color,1.0);
}`
export const mazeResidentShaders={compute,render}
export class ResidentMazeGpu{
 readonly count:number
 private readonly context:GPUCanvasContext
 private readonly device:GPUDevice
 private readonly settings:GPUBuffer
 private readonly walls:GPUBuffer
 private readonly paths:GPUBuffer
 private readonly states:[GPUBuffer,GPUBuffer]
 private readonly computePipeline:GPUComputePipeline
 private readonly renderPipeline:GPURenderPipeline
 private readonly computeGroups:[GPUBindGroup,GPUBindGroup]
 private readonly renderGroups:[GPUBindGroup,GPUBindGroup]
 private readonly cols:number
 private readonly rows:number
 private current=0
 private depth=0
 private disposed=false
 private constructor(maze:Maze,context:GPUCanvasContext,device:GPUDevice,buffers:GPUBuffer[],
  cp:GPUComputePipeline,rp:GPURenderPipeline,cg:GPUBindGroup[],rg:GPUBindGroup[]){
  this.count=maze.walls.length;this.cols=maze.columns;this.rows=maze.rows
  this.context=context;this.device=device;this.settings=buffers[0]!
  this.walls=buffers[1]!;this.paths=buffers[2]!
  this.states=[buffers[3]!,buffers[4]!];this.computePipeline=cp;this.renderPipeline=rp
  this.computeGroups=cg as [GPUBindGroup,GPUBindGroup]
  this.renderGroups=rg as [GPUBindGroup,GPUBindGroup]
 }
 static async create(canvas:HTMLCanvasElement,maze:Maze){
  const adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'})
  if(!adapter)throw new Error('real WebGPU required for maze')
  const device=await adapter.requestDevice()
  const context=canvas.getContext('webgpu') as GPUCanvasContext|null
  if(!context){device.destroy();throw new Error('WebGPU canvas unavailable')}
  const format=navigator.gpu.getPreferredCanvasFormat()
  context.configure({device,format,alphaMode:'opaque'})
  const count=maze.walls.length
  const buffers=[
   device.createBuffer({size:32,usage:U.UNIFORM|U.COPY_DST}),
   device.createBuffer({size:count*4,usage:U.STORAGE|U.COPY_DST}),
   device.createBuffer({size:count*4,usage:U.STORAGE|U.COPY_DST}),
   device.createBuffer({size:count*16,usage:U.STORAGE|U.COPY_SRC|U.COPY_DST}),
   device.createBuffer({size:count*16,usage:U.STORAGE|U.COPY_SRC|U.COPY_DST}),
  ]
  try{
   const modules=[]
   for(const text of [compute,render]){
     const shader=device.createShaderModule({code:text})
     const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
     if(errors.length)throw new Error('maze WGSL: '+errors.map(x=>x.message).join('; '))
     modules.push(shader)
   }
   const cp=await device.createComputePipelineAsync({layout:'auto',compute:{module:modules[0]!,entryPoint:'main'}})
   const rp=await device.createRenderPipelineAsync({layout:'auto',
     vertex:{module:modules[1]!,entryPoint:'vs'},
     fragment:{module:modules[1]!,entryPoint:'fs',targets:[{format}]},
     primitive:{topology:'triangle-list'}})
   const computeGroups=[0,1].map(i=>device.createBindGroup({
     layout:cp.getBindGroupLayout(0),entries:[
       {binding:0,resource:{buffer:buffers[0]!}},
       {binding:1,resource:{buffer:buffers[1]!}},
       {binding:2,resource:{buffer:buffers[3+i]!}},
       {binding:3,resource:{buffer:buffers[4-i]!}},
     ]}))
   const renderGroups=[0,1].map(i=>device.createBindGroup({
     layout:rp.getBindGroupLayout(0),entries:[
       {binding:0,resource:{buffer:buffers[0]!}},
       {binding:1,resource:{buffer:buffers[1]!}},
       {binding:2,resource:{buffer:buffers[3+i]!}},
       {binding:3,resource:{buffer:buffers[2]!}},
     ]}))
   const result=new ResidentMazeGpu(maze,context,device,buffers,cp,rp,computeGroups,renderGroups)
   const initial=beginSearch(maze),states=new Float32Array(count*4)
   for(let i=0;i<count;i++){
     states[i*4]=initial.visited[i]!
     states[i*4+1]=initial.frontier[i]!
     states[i*4+2]=initial.distance[i]!
   }
   device.queue.writeBuffer(buffers[1]!,0,Uint32Array.from(maze.walls))
   device.queue.writeBuffer(buffers[2]!,0,new Uint32Array(count))
   device.queue.writeBuffer(buffers[3]!,0,states)
   result.start=maze.start;result.goal=maze.goal
   result.updateSettings()
   return result
  }catch(error){for(const buffer of buffers)buffer.destroy();context.unconfigure();device.destroy();throw error}
 }
 private updateSettings(){
   const values=new Uint32Array(8)
   values[0]=this.cols;values[1]=this.rows;values[2]=this.depth
   values[3]=this.start;values[4]=this.goal
   this.device.queue.writeBuffer(this.settings,0,values)
 }
 private start=0
 private goal=0
 frame(){
   if(this.disposed)throw new Error('maze GPU destroyed')
   const settings=new Uint32Array([this.cols,this.rows,this.depth,this.start,this.goal,0,0,0])
   this.device.queue.writeBuffer(this.settings,0,settings)
   const encoder=this.device.createCommandEncoder()
   const pass=encoder.beginComputePass()
   pass.setPipeline(this.computePipeline);pass.setBindGroup(0,this.computeGroups[this.current])
   pass.dispatchWorkgroups(Math.ceil(this.count/128));pass.end()
   const next=1-this.current
   const display=encoder.beginRenderPass({colorAttachments:[{
      view:this.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',
      clearValue:{r:0.03,g:0.05,b:0.1,a:1},
   }]})
   display.setPipeline(this.renderPipeline);display.setBindGroup(0,this.renderGroups[next])
   display.draw(3);display.end()
   this.device.queue.submit([encoder.finish()])
   this.current=next;this.depth++
 }
 async read(){
   if(this.disposed)throw new Error('maze GPU destroyed')
   const bytes=this.count*16,staging=this.device.createBuffer({size:bytes,usage:U.MAP_READ|U.COPY_DST})
   try{
     const command=this.device.createCommandEncoder()
     command.copyBufferToBuffer(this.states[this.current],0,staging,0,bytes)
     this.device.queue.submit([command.finish()])
     await staging.mapAsync(M.READ)
     const result=new Float32Array(staging.getMappedRange().slice(0))
     staging.unmap();return result
   }finally{staging.destroy()}
 }
 setPath(path:number[]){const mask=new Uint32Array(this.count);for(const i of path)mask[i]=1
   this.device.queue.writeBuffer(this.paths,0,mask)}
 draw(){
   if(this.disposed)throw new Error('maze GPU destroyed')
   const encoder=this.device.createCommandEncoder()
   const pass=encoder.beginRenderPass({colorAttachments:[{
      view:this.context.getCurrentTexture().createView(),loadOp:'clear',storeOp:'store',
      clearValue:{r:0.03,g:0.05,b:0.1,a:1},
   }]})
   pass.setPipeline(this.renderPipeline);pass.setBindGroup(0,this.renderGroups[this.current])
   pass.draw(3);pass.end();this.device.queue.submit([encoder.finish()])
 }
 get steps(){return this.depth}
 destroy(){if(this.disposed)return;this.disposed=true;
  this.settings.destroy();this.walls.destroy();this.paths.destroy()
  for(const b of this.states)b.destroy();this.context.unconfigure();this.device.destroy()}
}

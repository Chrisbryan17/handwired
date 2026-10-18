import {type GpuAllocation} from '../../../src/runtime/frontiers-device.ts'
import {emitInlineCircuitWgsl} from '../../../src/runtime/webgpu-inline.ts'
import {SpatialGridGpu} from '../../../src/runtime/frontiers-spatial-gpu.ts'
import {type SpatialGridConfig} from '../../../src/runtime/frontiers-spatial.ts'
import {buildSphPressureCircuit,validateSph,
 type SphState,type SphSettings} from './model.ts'

/** True 3D weakly compressible SPH WebGPU solver.
 * Uses the independently verified GPU uniform spatial linked grid.
 * Density/pressure and forces are real numerical kernels, never vertex animation.
 * Does not yet implement free-surface mesh reconstruction or FLIP.
 */
const pressureCircuit=buildSphPressureCircuit()
const shared=`
struct Params {
 domainMin:vec4<f32>,
 domainMax:vec4<f32>,
 gridOrigin:vec4<f32>,
 dimensions:vec4<u32>,
 physics:vec4<f32>,
 material:vec4<f32>,
 extra:vec4<f32>,
 count:u32,padA:u32,padB:u32,padC:u32,
};
const PI:f32=3.141592653589793;
fn coords(p:vec3<f32>)->vec3<i32>{
 return vec3<i32>(floor((p-params.gridOrigin.xyz)/params.extra.y));
}
fn inside(c:vec3<i32>)->bool{
 return all(c>=vec3<i32>(0)) &&
  c.x<i32(params.dimensions.x)&&c.y<i32(params.dimensions.y)&&
  c.z<i32(params.dimensions.z);
}
fn cell(c:vec3<i32>)->u32{
 return (u32(c.z)*params.dimensions.y+u32(c.y))*params.dimensions.x+u32(c.x);
}
fn poly6(r2:f32)->f32{
 let h=params.physics.x;
 if(r2>=h*h){return 0.0;}
 let v=h*h-r2;
 return (315.0/(64.0*PI*pow(h,9.0)))*v*v*v;
}
`
export const sphDensityWgsl=`
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> heads:array<atomic<i32>>;
@group(0) @binding(2) var<storage,read> nextInCell:array<i32>;
@group(0) @binding(3) var<storage,read_write> densityPressure:array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params:Params;
`+shared+emitInlineCircuitWgsl(pressureCircuit.circuit,pressureCircuit.outputWires,'handwiredSphPressure')+`
@compute @workgroup_size(128)
fn density(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;
 if(i>=params.count){return;}
 let p=positions[i].xyz;
 let home=coords(p);
 if(!inside(home)){
  densityPressure[i]=vec4<f32>(0.0);return;
 }
 var rho=0.0;
 for(var z=-1;z<=1;z=z+1){
  for(var y=-1;y<=1;y=y+1){
   for(var x=-1;x<=1;x=x+1){
    let c=home+vec3<i32>(x,y,z);
    if(!inside(c)){continue;}
    var node=atomicLoad(&heads[cell(c)]);
    var visited=0u;
    loop{
     if(node<0||visited>=params.count){break;}
     let j=u32(node);
     let delta=p-positions[j].xyz;
     rho=rho+positions[j].w*poly6(dot(delta,delta));
     node=nextInCell[j];
     visited=visited+1u;
    }
   }
  }
 }
 var input:array<f32,3>;
 input[0u]=rho;input[1u]=params.physics.w;input[2u]=params.physics.z;
 let pSigned=handwiredSphPressure(input);
 densityPressure[i]=vec4<f32>(rho,max(0.0,pSigned[0u]),0.0,0.0);
}
`
export const sphIntegrateWgsl=`
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> heads:array<atomic<i32>>;
@group(0) @binding(3) var<storage,read> nextInCell:array<i32>;
@group(0) @binding(4) var<storage,read> densityPressure:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> outPositions:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> outVelocities:array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params:Params;
`+shared+`
@compute @workgroup_size(128)
fn forceAndAdvance(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;
 if(i>=params.count){return;}
 let particle=positions[i];
 let speed=velocities[i];
 let home=coords(particle.xyz);
 let rhoI=max(densityPressure[i].x,1.0e-7);
 let pressureI=densityPressure[i].y;
 let h=params.physics.x;
 var accel=vec3<f32>(params.material.z,params.material.w,params.extra.x);
 if(inside(home)){
  for(var z=-1;z<=1;z=z+1){
   for(var y=-1;y<=1;y=y+1){
    for(var x=-1;x<=1;x=x+1){
     let c=home+vec3<i32>(x,y,z);
     if(!inside(c)){continue;}
     var node=atomicLoad(&heads[cell(c)]);
     var visited=0u;
     loop{
      if(node<0||visited>=params.count){break;}
      let j=u32(node);
      if(j!=i){
       let other=positions[j];
       let delta=particle.xyz-other.xyz;
       let r=length(delta);
       if(r<h&&r>1.0e-8){
        let rhoJ=max(densityPressure[j].x,1.0e-7);
        let pressureJ=densityPressure[j].y;
        let gradient=-45.0/(PI*pow(h,6.0))*pow(h-r,2.0);
        let coefficient=-other.w*(pressureI/(rhoI*rhoI)+
         pressureJ/(rhoJ*rhoJ))*gradient/r;
        let laplacian=45.0/(PI*pow(h,6.0))*(h-r);
        accel=accel+coefficient*delta+
         (params.material.x*other.w*laplacian/rhoJ)*(velocities[j].xyz-speed.xyz);
       }
      }
      node=nextInCell[j];
      visited=visited+1u;
     }
    }
   }
  }
 }
 let dt=params.physics.y;
 var updatedVelocity=speed.xyz+dt*accel;
 var updatedPosition=particle.xyz+dt*updatedVelocity;
 for(var axis=0u;axis<3u;axis=axis+1u){
  if(updatedPosition[axis]<params.domainMin[axis]){
   updatedPosition[axis]=params.domainMin[axis];
   updatedVelocity[axis]=-updatedVelocity[axis]*params.material.y;
  }else if(updatedPosition[axis]>params.domainMax[axis]){
   updatedPosition[axis]=params.domainMax[axis];
   updatedVelocity[axis]=-updatedVelocity[axis]*params.material.y;
  }
 }
 outPositions[i]=vec4<f32>(updatedPosition,particle.w);
 outVelocities[i]=vec4<f32>(updatedVelocity,0.0);
}
`
const usage={STORAGE:128,COPY_DST:8,COPY_SRC:4,UNIFORM:64,MAP_READ:1}
export class CataclysmSphGpu {
 readonly grid:SpatialGridGpu
 readonly gpu:GpuAllocation
 readonly count:number
 get velocityBuffer():GPUBuffer{this.ensure();return this.velocities}
 readonly settings:SphSettings
 private readonly velocities:GPUBuffer
 private readonly density:GPUBuffer
 private readonly outputPositions:GPUBuffer
 private readonly outputVelocities:GPUBuffer
 private readonly densityPipeline:GPUComputePipeline
 private readonly physicsPipeline:GPUComputePipeline
 private readonly densityBind:GPUBindGroup
 private readonly physicsBind:GPUBindGroup
 private ticks=0
 private disposed=false
 private constructor(grid:SpatialGridGpu,settings:SphSettings,
  velocities:GPUBuffer,density:GPUBuffer,
  outputPositions:GPUBuffer,outputVelocities:GPUBuffer,
  densityPipeline:GPUComputePipeline,physicsPipeline:GPUComputePipeline,
  densityBind:GPUBindGroup,physicsBind:GPUBindGroup){
   this.grid=grid;this.gpu=grid.gpu;this.count=grid.pointCount
   this.settings=settings;this.velocities=velocities;this.density=density
   this.outputPositions=outputPositions;this.outputVelocities=outputVelocities
   this.densityPipeline=densityPipeline;this.physicsPipeline=physicsPipeline
   this.densityBind=densityBind;this.physicsBind=physicsBind
 }
 static async create(state:SphState,p:SphSettings):Promise<CataclysmSphGpu>{
  const count=validateSph(state,p,131_072)
  const cellSize=Math.max(p.h,.5)
  const dims=p.min.map((min,i)=>Math.ceil((p.max[i]!-min)/cellSize)+1)
  const spatial:SpatialGridConfig={origin:p.min,dimensions:dims as unknown as [number,number,number],
    cellSize,radius:p.h}
  const grid=await SpatialGridGpu.create(state.positions,spatial)
  try{
   const device=grid.gpu.device
   const alloc=(bytes:number,flags:number,label:string)=>grid.gpu.allocate(bytes,flags,label)
   const velocities=alloc(count*16,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,'SPH particle velocity')
   const density=alloc(count*16,usage.STORAGE|usage.COPY_SRC,'SPH density and equation-of-state pressure')
   const outputPositions=alloc(count*16,usage.STORAGE|usage.COPY_SRC,'SPH predicted positions')
   const outputVelocities=alloc(count*16,usage.STORAGE|usage.COPY_SRC,'SPH new momentum')
   const uniform=alloc(128,usage.UNIFORM|usage.COPY_DST,'SPH kernel and boundary parameters')
   device.queue.writeBuffer(velocities,0,state.velocities)
   const buffer=new ArrayBuffer(128),v=new DataView(buffer)
   for(let i=0;i<3;i++){
    v.setFloat32(i*4,p.min[i]!,true)
    v.setFloat32(16+i*4,p.max[i]!,true)
    v.setFloat32(32+i*4,spatial.origin[i]!,true)
    v.setUint32(48+i*4,spatial.dimensions[i]!,true)
   }
   for(let i=0;i<4;i++){
    v.setFloat32(64+i*4,[p.h,p.dt,p.restDensity,p.stiffness][i]!,true)
    v.setFloat32(80+i*4,[p.viscosity,p.boundaryRestitution,p.gravity[0],p.gravity[1]][i]!,true)
   }
   v.setFloat32(96,p.gravity[2],true);v.setFloat32(100,cellSize,true)
   v.setUint32(112,count,true)
   device.queue.writeBuffer(uniform,0,buffer)
   const compile=(code:string,label:string)=>device.createShaderModule({code,label})
   const densityShader=compile(sphDensityWgsl,'Handwired 3D SPH density and pressure')
   const physicsShader=compile(sphIntegrateWgsl,'3D SPH forces and tank bounds')
   for(const shader of [densityShader,physicsShader]){
    const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
    if(errors.length)throw new Error('SPH WGSL: '+errors.map(x=>x.message).join('; '))
   }
   const densityPipeline=await device.createComputePipelineAsync({
    layout:'auto',compute:{module:densityShader,entryPoint:'density'}})
   const physicsPipeline=await device.createComputePipelineAsync({
    layout:'auto',compute:{module:physicsShader,entryPoint:'forceAndAdvance'}})
   const binding=(pipeline:GPUComputePipeline,binds:GPUBuffer[])=>device.createBindGroup({
    layout:pipeline.getBindGroupLayout(0),
    entries:binds.map((buffer,binding)=>({binding,resource:{buffer}}))
   })
   const densityBind=binding(densityPipeline,[grid.positions,grid.cellHeads,grid.next,density,uniform])
   const physicsBind=binding(physicsPipeline,[grid.positions,velocities,
    grid.cellHeads,grid.next,density,outputPositions,outputVelocities,uniform])
   return new CataclysmSphGpu(grid,p,velocities,density,outputPositions,outputVelocities,
    densityPipeline,physicsPipeline,densityBind,physicsBind)
  }catch(e){grid.dispose();throw e}
 }
 get stepCount(){return this.ticks}
 private ensure(){if(this.disposed)throw new Error('SPH solver disposed');this.gpu.check()}
 encodeStep(encoder:GPUCommandEncoder){
  this.ensure()
  this.grid.encodeBuild(encoder)
  let pass=encoder.beginComputePass({label:'3D SPH density + Handwired equation of state'})
  pass.setPipeline(this.densityPipeline);pass.setBindGroup(0,this.densityBind)
  pass.dispatchWorkgroups(Math.ceil(this.count/128));pass.end()
  pass=encoder.beginComputePass({label:'SPH pressure viscosity and boundary physics'})
  pass.setPipeline(this.physicsPipeline);pass.setBindGroup(0,this.physicsBind)
  pass.dispatchWorkgroups(Math.ceil(this.count/128));pass.end()
  encoder.copyBufferToBuffer(this.outputPositions,0,this.grid.positions,0,this.count*16)
  encoder.copyBufferToBuffer(this.outputVelocities,0,this.velocities,0,this.count*16)
  this.ticks++
 }
 step(){
  this.ensure()
  const encoder=this.gpu.device.createCommandEncoder()
  this.encodeStep(encoder)
  this.gpu.device.queue.submit([encoder.finish()])
 }
 async flush(){this.ensure();await this.gpu.device.queue.onSubmittedWorkDone();this.ensure()}
 /** Diagnostic readback of the *previous* compute step's GPU density and
  * Handwired-compiled pressure field. Never called during ordinary stepping.
  */
 async readDensityPressure():Promise<{density:Float32Array,pressure:Float32Array}>{
  this.ensure()
  const device=this.gpu.device
  const size=this.count*16
  const stage=device.createBuffer({size,usage:usage.COPY_DST|usage.MAP_READ})
  try{
   const encoder=device.createCommandEncoder()
   encoder.copyBufferToBuffer(this.density,0,stage,0,size)
   device.queue.submit([encoder.finish()])
   await stage.mapAsync(1)
   const packed=new Float32Array(stage.getMappedRange().slice(0))
   stage.unmap()
   const density=new Float32Array(this.count)
   const pressure=new Float32Array(this.count)
   for(let i=0;i<this.count;i++){
    density[i]=packed[i*4]!
    pressure[i]=packed[i*4+1]!
   }
   this.ensure()
   return {density,pressure}
  }finally{stage.destroy()}
 }
 async read():Promise<SphState>{
  this.ensure()
  const device=this.gpu.device,size=this.count*16
  const buffers=[this.outputPositions,this.velocities]
  const stages=buffers.map(()=>device.createBuffer({size,
   usage:usage.COPY_DST|usage.MAP_READ}))
  try{
   const encoder=device.createCommandEncoder()
   for(let i=0;i<buffers.length;i++)
    encoder.copyBufferToBuffer(buffers[i]!,0,stages[i]!,0,size)
   device.queue.submit([encoder.finish()])
   await Promise.all(stages.map(b=>b.mapAsync(1)))
   const fields=stages.map(b=>new Float32Array(b.getMappedRange().slice(0)))
   for(const b of stages)b.unmap()
   this.ensure()
   return {positions:fields[0]!,velocities:fields[1]!}
  }finally{for(const b of stages)b.destroy()}
 }
 dispose(){if(this.disposed)return;this.disposed=true;this.grid.dispose()}
}

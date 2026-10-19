import {requestStrictGpu,type GpuAllocation} from '../../../src/runtime/frontiers-device.ts'
import {emitInlineCircuitWgsl} from '../../../src/runtime/webgpu-inline.ts'
import {buildLegMotorCircuit,validateLegParameters,validateLegPopulation,
 type PlanarLegParameters,type PlanarLegState} from './model.ts'

/** Real batched articulated 2-link planar dynamics using one GPU
 * work-item per environment. Hip pinned, no inter-env coupling.
 * Controller is a compiled Handwired PD circuit, not learned inference.
 */
const controller=buildLegMotorCircuit()
export const batchedLegWgsl=`
struct Parameters {
 geometry:vec4<f32>,
 physical:vec4<f32>,
 ground:vec4<f32>,
 control:vec4<f32>,
 targets:vec4<f32>,
 count:u32,pad0:u32,pad1:u32,pad2:u32,
};
@group(0) @binding(0) var<storage,read> current:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> next:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p:Parameters;
`+emitInlineCircuitWgsl(controller.circuit,controller.outputWires,'handwiredLegMotor')+`
fn motor(q:f32,w:f32,desired:f32)->f32 {
 var input:array<f32,5>;
 input[0u]=q;input[1u]=w;input[2u]=desired;
 input[3u]=p.control.x;input[4u]=p.control.y;
 let generated=handwiredLegMotor(input);
 return clamp(generated[0u],-p.control.z,p.control.z);
}
@compute @workgroup_size(128)
fn advanceLegs(@builtin(global_invocation_id) tid:vec3<u32>){
 let i=tid.x;
 if(i>=p.count){return;}
 let s=current[i];
 let q1=s.x;
 let q2=s.y;
 let w1=s.z;
 let w2=s.w;
 let l1=p.geometry.x;
 let l2=p.geometry.y;
 let m1=p.geometry.z;
 let m2=p.geometry.w;
 let angle=q1+q2;
 let s1=sin(q1);
 let c1=cos(q1);
 let sa=sin(angle);
 let ca=cos(angle);
 let footY=p.physical.y-l1*c1-l2*ca;
 let jx1=l1*c1+l2*ca;
 let jx2=l2*ca;
 let jy1=l1*s1+l2*sa;
 let jy2=l2*sa;
 let footVx=jx1*w1+jx2*w2;
 let footVy=jy1*w1+jy2*w2;
 let penetration=max(0.0,p.physical.z-footY);
 var normalForce=0.0;
 if(penetration>0.0){
  normalForce=max(0.0,p.ground.x*penetration-p.ground.y*footVy);
 }
 let frictionForce=clamp(-p.ground.w*footVx,
  -p.ground.z*normalForce,p.ground.z*normalForce);
 let m11=(m1+m2)*l1*l1+m2*l2*l2+2.0*m2*l1*l2*cos(q2);
 let m12=m2*l2*l2+m2*l1*l2*cos(q2);
 let m22=m2*l2*l2;
 let det=m11*m22-m12*m12;
 let h=-m2*l1*l2*sin(q2);
 let coriolis1=h*(2.0*w1*w2+w2*w2);
 let coriolis2=-h*w1*w1;
 let grav1=p.physical.x*((m1+m2)*l1*s1+m2*l2*sa);
 let grav2=p.physical.x*m2*l2*sa;
 let external1=jx1*frictionForce+jy1*normalForce;
 let external2=jx2*frictionForce+jy2*normalForce;
 let rhs1=motor(q1,w1,p.targets.x)+external1-coriolis1-grav1-p.control.w*w1;
 let rhs2=motor(q2,w2,p.targets.y)+external2-coriolis2-grav2-p.control.w*w2;
 let acceleration1=(m22*rhs1-m12*rhs2)/det;
 let acceleration2=(m11*rhs2-m12*rhs1)/det;
 let v1=w1+p.physical.w*acceleration1;
 let v2=w2+p.physical.w*acceleration2;
 let nextQ1=q1+p.physical.w*v1;
 let nextQ2=q2+p.physical.w*v2;
 next[i]=vec4<f32>(nextQ1,nextQ2,v1,v2);
}
`
const usage={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64,MAP_READ:1}
export class BatchedLegsGpu {
 readonly gpu:GpuAllocation
 readonly count:number
 readonly parameters:PlanarLegParameters
 private readonly buffers:readonly [GPUBuffer,GPUBuffer]
 private readonly pipeline:GPUComputePipeline
 private readonly binds:readonly [GPUBindGroup,GPUBindGroup]
 private active=0
 private ticks=0
 private disposed=false
 private constructor(gpu:GpuAllocation,count:number,p:PlanarLegParameters,
  buffers:readonly [GPUBuffer,GPUBuffer],pipeline:GPUComputePipeline,
  binds:readonly [GPUBindGroup,GPUBindGroup]){
  this.gpu=gpu;this.count=count;this.parameters=p
  this.buffers=buffers;this.pipeline=pipeline;this.binds=binds
 }
 static async create(initial:PlanarLegState,
  p:PlanarLegParameters):Promise<BatchedLegsGpu>{
  const count=validateLegPopulation(initial,65536)
  validateLegParameters(p)
  const gpu=await requestStrictGpu('Neural Machines: batched articulated leg GPU')
  try{
   const device=gpu.device
   if(Math.ceil(count/128)>device.limits.maxComputeWorkgroupsPerDimension)
    throw new Error('GPU articulated environment workgroup limit exceeded')
   const a=gpu.allocate(count*16,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,
    'Articulated leg joints current state')
   const b=gpu.allocate(count*16,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,
    'Articulated leg joints next state')
   device.queue.writeBuffer(a,0,initial.joints)
   const uniform=gpu.allocate(96,usage.UNIFORM|usage.COPY_DST,
    'Two-link robot physical parameters')
   const data=new ArrayBuffer(96),view=new DataView(data)
   const fields=[
    p.length1,p.length2,p.mass1,p.mass2,
    p.gravity,p.hipHeight,p.groundHeight,p.dt,
    p.stiffness,p.contactDamping,p.friction,p.tangentDamping,
    p.kp,p.kd,p.torqueLimit,p.jointDamping,
    p.targetHip,p.targetKnee,0,0,
   ]
   for(let i=0;i<fields.length;i++)
    view.setFloat32(i*4,fields[i]!,true)
   view.setUint32(80,count,true)
   device.queue.writeBuffer(uniform,0,data)
   const shader=device.createShaderModule({code:batchedLegWgsl,
    label:'Actual Handwired-generated robot controller and articulated 2DOF dynamics'})
   const errors=(await shader.getCompilationInfo()).messages.filter(x=>x.type==='error')
   if(errors.length)throw new Error('Robotics WGSL: '+errors.map(x=>x.message).join('; '))
   const pipeline=await device.createComputePipelineAsync({
    layout:'auto',compute:{module:shader,entryPoint:'advanceLegs'}})
   const binding=(current:GPUBuffer,next:GPUBuffer)=>device.createBindGroup({
    layout:pipeline.getBindGroupLayout(0),entries:[
     {binding:0,resource:{buffer:current}},
     {binding:1,resource:{buffer:next}},
     {binding:2,resource:{buffer:uniform}},
    ]})
   return new BatchedLegsGpu(gpu,count,p,[a,b],pipeline,
    [binding(a,b),binding(b,a)])
  }catch(e){gpu.dispose();throw e}
 }
 get steps(){return this.ticks}
 get residentJointBuffer(){
  this.ensure()
  return this.buffers[this.active]!
 }
 private ensure(){
  if(this.disposed)throw new Error('Batched robot GPU solver disposed')
  this.gpu.check()
 }
 encodeStep(encoder:GPUCommandEncoder){
  this.ensure()
  const pass=encoder.beginComputePass({label:'True GPU articulated dynamics and Handwired motor control'})
  pass.setPipeline(this.pipeline)
  pass.setBindGroup(0,this.binds[this.active]!)
  pass.dispatchWorkgroups(Math.ceil(this.count/128))
  pass.end()
  this.active=1-this.active
  this.ticks++
 }
 step(){
  const encoder=this.gpu.device.createCommandEncoder()
  this.encodeStep(encoder)
  this.gpu.device.queue.submit([encoder.finish()])
 }
 async flush(){this.ensure();await this.gpu.device.queue.onSubmittedWorkDone();this.ensure()}
 async read():Promise<PlanarLegState>{
  this.ensure()
  const device=this.gpu.device
  const bytes=this.count*16
  const stage=device.createBuffer({size:bytes,
   usage:usage.COPY_DST|usage.MAP_READ})
  try{
   const encoder=device.createCommandEncoder()
   encoder.copyBufferToBuffer(this.residentJointBuffer,0,stage,0,bytes)
   device.queue.submit([encoder.finish()])
   await stage.mapAsync(1)
   const joints=new Float32Array(stage.getMappedRange().slice(0))
   stage.unmap()
   this.ensure()
   return {joints}
  }finally{stage.destroy()}
 }
 dispose(){if(this.disposed)return;this.disposed=true;this.gpu.dispose()}
}

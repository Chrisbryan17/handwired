import {requestStrictGpu,type GpuAllocation} from '../../../src/runtime/frontiers-device.ts'
import {emitInlineCircuitWgsl} from '../../../src/runtime/webgpu-inline.ts'
import {Circuit} from '../../../src/runtime/numeric-ui-core.ts'
import {validateBonded,validateStructuralSettings,type BondedBody,
  type BondedSettings} from './fracture-model.ts'

/** Bonded 3D structure GPU solver, fixed topology and fracture state.
 * Every physics step evaluates spring strain/damage on actual GPU workgroups,
 * and a CSR gather integrates nodes from per-bond forces. No CPU fallback.
 */
function buildStructuralIntegrator(){
 const c=new Circuit(6)
 const src=Array.from({length:6},(_,i)=>c.source(i))
 const forceAccel=c.multiply(src[2]!,src[3]!)
 const accel=c.roundAffine([[forceAccel,1],[src[5]!,1]])
 const deltaVelocity=c.multiply(accel,src[4]!)
 const vNext=c.roundAffine([[src[1]!,1],[deltaVelocity,1]])
 const deltaPosition=c.multiply(vNext,src[4]!)
 const pNext=c.roundAffine([[src[0]!,1],[deltaPosition,1]])
 return {c,outputs:[pNext,vNext]}
}
const integrator=buildStructuralIntegrator()
export const bondedForceWgsl=`
struct Params {
  nodes:u32,bonds:u32,pad0:u32,pad1:u32,
  timestep:vec4<f32>,
};
@group(0) @binding(0) var<storage,read> nodePositions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> nodeVelocities:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> endpoints:array<vec2<u32>>;
@group(0) @binding(3) var<storage,read> material:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> fracture:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> linkForce:array<vec4<f32>>;
@group(0) @binding(6) var<uniform> params:Params;
@compute @workgroup_size(128)
fn bondUpdate(@builtin(global_invocation_id) id:vec3<u32>){
 let index=id.x;
 if(index>=params.bonds){return;}
 let endpointsPair=endpoints[index];
 let a=endpointsPair.x;let b=endpointsPair.y;
 let pa=nodePositions[a].xyz,pb=nodePositions[b].xyz;
 let va=nodeVelocities[a].xyz,vb=nodeVelocities[b].xyz;
 let axis=pb-pa;
 let dist=length(axis);
 let props=material[index];
 var state=fracture[index];
 if(state.w<.5){
  linkForce[index]=vec4<f32>(0.0);
  return;
 }
 let extension=dist-props.x;
 let strain=abs(extension)/props.x;
 if(strain>props.w){
  let increment=params.timestep.x*state.y*
    (strain-props.w)/max(1.0e-8,state.x-props.w);
  state.z=min(1.0,state.z+increment);
 }
 if(strain>=state.x || state.z>=1.0){
  state.w=0.0;
  fracture[index]=state;
  linkForce[index]=vec4<f32>(0.0);
  return;
 }
 fracture[index]=state;
 if(dist<=1.0e-12){
  linkForce[index]=vec4<f32>(0.0);
  return;
 }
 let direction=axis/dist;
 let relativeSpeed=dot(vb-va,direction);
 let magnitude=props.y*extension+props.z*relativeSpeed;
 linkForce[index]=vec4<f32>(magnitude*direction,0.0);
}
`
export const bondedIntegrateWgsl=`
struct Params {
  nodes:u32,bonds:u32,pad0:u32,pad1:u32,
  timestep:vec4<f32>,
};
@group(0) @binding(0) var<storage,read> nodePositions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> nodeVelocities:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> adjacencyOffset:array<u32>;
@group(0) @binding(3) var<storage,read> adjacency:array<vec2<u32>>;
@group(0) @binding(4) var<storage,read> linkForce:array<vec4<f32>>;
@group(0) @binding(5) var<storage,read_write> outputPositions:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> outputVelocities:array<vec4<f32>>;
@group(0) @binding(7) var<uniform> params:Params;
`+emitInlineCircuitWgsl(integrator.c,integrator.outputs,'handwiredStructuralStep')+`
@compute @workgroup_size(128)
fn advanceNodes(@builtin(global_invocation_id) id:vec3<u32>){
 let node=id.x;
 if(node>=params.nodes){return;}
 let oldPos=nodePositions[node];
 let oldVel=nodeVelocities[node];
 if(oldPos.w<=0.0){
  outputPositions[node]=oldPos;
  outputVelocities[node]=oldVel;
  return;
 }
 var force=vec3<f32>(0.0);
 for(var i=adjacencyOffset[node];i<adjacencyOffset[node+1u];i=i+1u){
  let item=adjacency[i];
  let impulse=linkForce[item.x].xyz;
  force=force+select(impulse,-impulse,item.y==1u);
 }
 var outPosition=vec3<f32>(0.0);
 var outVelocity=vec3<f32>(0.0);
 for(var axis=0u;axis<3u;axis=axis+1u){
  var input:array<f32,6>;
  input[0u]=oldPos[axis];
  input[1u]=oldVel[axis];
  input[2u]=force[axis];
  input[3u]=oldPos.w;
  input[4u]=params.timestep.x;
  input[5u]=params.timestep[axis+1u];
  let result=handwiredStructuralStep(input);
  outPosition[axis]=result[0u];
  outVelocity[axis]=result[1u];
 }
 outputPositions[node]=vec4<f32>(outPosition,oldPos.w);
 outputVelocities[node]=vec4<f32>(outVelocity,0.0);
}
`

const usage={STORAGE:128,COPY_DST:8,COPY_SRC:4,UNIFORM:64,MAP_READ:1}
export const bondedImpactWgsl=`
struct Impulse { node:u32,padA:u32,padB:u32,padC:u32,vector:vec4<f32> };
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> velocities:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> impulse:Impulse;
@compute @workgroup_size(1)
fn applyImpulse(){
 let i=impulse.node;
 if(i>=arrayLength(&positions)){return;}
 let inv=positions[i].w;
 let prev=velocities[i];
 velocities[i]=vec4<f32>(prev.xyz+inv*impulse.vector.xyz,prev.w);
}
`

export class BondedStructureGpu{
 readonly gpu:GpuAllocation
 readonly nodeCount:number
 readonly bondCount:number
 readonly settings:BondedSettings
 readonly nodePositions:GPUBuffer
 readonly nodeVelocities:GPUBuffer
 readonly fractureState:GPUBuffer
 readonly bondEndpoints:GPUBuffer
 private impactProgram:GPUComputePipeline|null=null
 private impactBuffer:GPUBuffer|null=null
 private impactBindings:GPUBindGroup|null=null
 private readonly outPos:GPUBuffer
 private readonly outVel:GPUBuffer
 private readonly updateBonds:GPUComputePipeline
 private readonly updateNodes:GPUComputePipeline
 private readonly bondsBind:GPUBindGroup
 private readonly nodesBind:GPUBindGroup
 private readonly initial:BondedBody
 private disposed=false
 private frames=0
 private constructor(gpu:GpuAllocation,initial:BondedBody,params:BondedSettings,
  pos:GPUBuffer,vel:GPUBuffer,fractureState:GPUBuffer,bondEndpoints:GPUBuffer,
  outPos:GPUBuffer,outVel:GPUBuffer,
  updateBonds:GPUComputePipeline,updateNodes:GPUComputePipeline,
  bondsBind:GPUBindGroup,nodesBind:GPUBindGroup){
   this.gpu=gpu;this.initial=initial;this.settings=params
   this.nodeCount=initial.positions.length/4;this.bondCount=initial.endpoints.length/2
   this.nodePositions=pos;this.nodeVelocities=vel;this.fractureState=fractureState
   this.bondEndpoints=bondEndpoints
   this.outPos=outPos;this.outVel=outVel
   this.updateBonds=updateBonds;this.updateNodes=updateNodes
   this.bondsBind=bondsBind;this.nodesBind=nodesBind
 }
 static async create(initial:BondedBody,params:BondedSettings):Promise<BondedStructureGpu>{
  const {nodes,bonds}=validateBonded(initial)
  validateStructuralSettings(params)
  const gpu=await requestStrictGpu('Handwired bonded lattice fracture')
  try{
   const device=gpu.device
   if(Math.ceil(Math.max(nodes,bonds)/128)>device.limits.maxComputeWorkgroupsPerDimension)
     throw new Error('Bonded structure exceeds device dispatch limit')
   const alloc=(bytes:number,flags:number,label:string)=>gpu.allocate(bytes,flags,label)
   const pos=alloc(nodes*16,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,'Bonded node centers')
   const vel=alloc(nodes*16,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,'Bonded node velocities')
   const ends=alloc(bonds*8,usage.STORAGE|usage.COPY_DST,'Bond topology')
   const mat=alloc(bonds*16,usage.STORAGE|usage.COPY_DST,'Axial stiffness/yield')
   const damage=alloc(bonds*16,usage.STORAGE|usage.COPY_DST|usage.COPY_SRC,'Irreversible fracture state')
   const forces=alloc(bonds*16,usage.STORAGE,'Axial forces per link')
   const outPos=alloc(nodes*16,usage.STORAGE|usage.COPY_SRC,'Integrated structural nodes')
   const outVel=alloc(nodes*16,usage.STORAGE|usage.COPY_SRC,'Integrated structural velocities')
   const offsets=new Uint32Array(nodes+1),degrees=new Uint32Array(nodes)
   for(let i=0;i<bonds;i++){
    degrees[initial.endpoints[i*2]!]!++
    degrees[initial.endpoints[i*2+1]!]!++
   }
   for(let i=0;i<nodes;i++)offsets[i+1]=offsets[i]!+degrees[i]!
   const ptr=offsets.slice(0,nodes)
   const adjacency=new Uint32Array(bonds*4)
   for(let i=0;i<bonds;i++){
    const a=initial.endpoints[i*2]!,b=initial.endpoints[i*2+1]!
    const left=ptr[a]!++,right=ptr[b]!++
    adjacency.set([i,0],left*2)
    adjacency.set([i,1],right*2)
   }
   const csr=alloc((nodes+1)*4,usage.STORAGE|usage.COPY_DST,'Node CSR offsets')
   const links=alloc(bonds*16,usage.STORAGE|usage.COPY_DST,'Directed adjacency edges')
   const uniforms=alloc(32,usage.UNIFORM|usage.COPY_DST,'Fracture timestep and gravity')
   const material=new Float32Array(bonds*4),states=new Float32Array(bonds*4)
   for(let i=0;i<bonds;i++){
    material.set(initial.bonds.subarray(i*8,i*8+4),i*4)
    states.set(initial.bonds.subarray(i*8+4,i*8+8),i*4)
   }
   device.queue.writeBuffer(pos,0,initial.positions)
   device.queue.writeBuffer(vel,0,initial.velocities)
   device.queue.writeBuffer(ends,0,initial.endpoints)
   device.queue.writeBuffer(mat,0,material)
   device.queue.writeBuffer(damage,0,states)
   device.queue.writeBuffer(csr,0,offsets)
   device.queue.writeBuffer(links,0,adjacency)
   const uniformData=new ArrayBuffer(32),view=new DataView(uniformData)
   view.setUint32(0,nodes,true);view.setUint32(4,bonds,true)
   for(let i=0;i<4;i++)view.setFloat32(16+i*4,[params.dt,...params.gravity][i]!,true)
   device.queue.writeBuffer(uniforms,0,uniformData)
   const compiled=(code:string,label:string)=>device.createShaderModule({code,label})
   const first=compiled(bondedForceWgsl,'Cohesive strain damage and Hookean bond forces')
   const second=compiled(bondedIntegrateWgsl,'Handwired structural integration')
   for(const mod of [first,second]){
    const issues=(await mod.getCompilationInfo()).messages.filter(x=>x.type==='error')
    if(issues.length)throw new Error('Structural WGSL: '+issues.map(x=>x.message).join('; '))
   }
   const bondPipeline=await device.createComputePipelineAsync({
    layout:'auto',compute:{module:first,entryPoint:'bondUpdate'}})
   const nodePipeline=await device.createComputePipelineAsync({
    layout:'auto',compute:{module:second,entryPoint:'advanceNodes'}})
   const bind=(pipeline:GPUComputePipeline,buffers:GPUBuffer[])=>device.createBindGroup({
    layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({
     binding,resource:{buffer}
    }))})
   const bondsBind=bind(bondPipeline,[pos,vel,ends,mat,damage,forces,uniforms])
   const nodesBind=bind(nodePipeline,[pos,vel,csr,links,forces,outPos,outVel,uniforms])
   return new BondedStructureGpu(gpu,initial,params,pos,vel,damage,ends,outPos,outVel,
    bondPipeline,nodePipeline,bondsBind,nodesBind)
  }catch(error){gpu.dispose();throw error}
 }
 get stepCount(){return this.frames}
 private ensure(){if(this.disposed)throw new Error('Bonded solver is disposed');this.gpu.check()}
 encodeStep(encoder:GPUCommandEncoder){
  this.ensure()
  let pass=encoder.beginComputePass({label:'Bond stress and irreversible fracture'})
  pass.setPipeline(this.updateBonds);pass.setBindGroup(0,this.bondsBind)
  pass.dispatchWorkgroups(Math.ceil(this.bondCount/128));pass.end()
  pass=encoder.beginComputePass({label:'Handwired-driven nodal dynamics'})
  pass.setPipeline(this.updateNodes);pass.setBindGroup(0,this.nodesBind)
  pass.dispatchWorkgroups(Math.ceil(this.nodeCount/128));pass.end()
  encoder.copyBufferToBuffer(this.outPos,0,this.nodePositions,0,this.nodeCount*16)
  encoder.copyBufferToBuffer(this.outVel,0,this.nodeVelocities,0,this.nodeCount*16)
  this.frames++
 }
 step(){
  const encoder=this.gpu.device.createCommandEncoder()
  this.encodeStep(encoder)
  this.gpu.device.queue.submit([encoder.finish()])
 }
 async flush(){this.ensure();await this.gpu.device.queue.onSubmittedWorkDone();this.ensure()}
 /** A user-triggered impulse applied directly to the live GPU velocity
  * state. Impact timing/targets are interactive, not scripted collapse.
  */
 applyImpulse(node:number,vector:readonly [number,number,number]){
  this.ensure()
  if(!Number.isSafeInteger(node)||node<0||node>=this.nodeCount ||
    !vector.every(Number.isFinite)||vector.some(x=>Math.abs(x)>500))
    throw new Error('invalid structural GPU impact')
  if(this.impactProgram===null){
   const device=this.gpu.device
   const module=device.createShaderModule({code:bondedImpactWgsl,
    label:'GPU interactive impulse on bonded structural node'})
   this.impactProgram=device.createComputePipeline({
    layout:'auto',compute:{module,entryPoint:'applyImpulse'}})
   this.impactBuffer=this.gpu.allocate(32,usage.UNIFORM|usage.COPY_DST,'User-directed impact')
   this.impactBindings=device.createBindGroup({
    layout:this.impactProgram.getBindGroupLayout(0),
    entries:[
     {binding:0,resource:{buffer:this.nodePositions}},
     {binding:1,resource:{buffer:this.nodeVelocities}},
     {binding:2,resource:{buffer:this.impactBuffer}},
    ]})
  }
  const data=new ArrayBuffer(32),view=new DataView(data)
  view.setUint32(0,node,true)
  for(let i=0;i<3;i++)view.setFloat32(16+i*4,vector[i]!,true)
  this.gpu.device.queue.writeBuffer(this.impactBuffer!,0,data)
  const encoder=this.gpu.device.createCommandEncoder()
  const pass=encoder.beginComputePass({label:'Interactive structural GPU impact'})
  pass.setPipeline(this.impactProgram)
  pass.setBindGroup(0,this.impactBindings)
  pass.dispatchWorkgroups(1)
  pass.end()
  this.gpu.device.queue.submit([encoder.finish()])
 }
 async read():Promise<BondedBody>{
  this.ensure()
  const dev=this.gpu.device
  const pLen=this.nodeCount*16,bLen=this.bondCount*16
  const staged=[pLen,pLen,bLen].map(size=>dev.createBuffer({size,
   usage:usage.COPY_DST|usage.MAP_READ}))
  try{
   const encoder=dev.createCommandEncoder()
   encoder.copyBufferToBuffer(this.nodePositions,0,staged[0]!,0,pLen)
   encoder.copyBufferToBuffer(this.nodeVelocities,0,staged[1]!,0,pLen)
   encoder.copyBufferToBuffer(this.fractureState,0,staged[2]!,0,bLen)
   dev.queue.submit([encoder.finish()])
   await Promise.all(staged.map(b=>b.mapAsync(1)))
   const positions=new Float32Array(staged[0]!.getMappedRange().slice(0))
   const velocities=new Float32Array(staged[1]!.getMappedRange().slice(0))
   const states=new Float32Array(staged[2]!.getMappedRange().slice(0))
   for(const b of staged)b.unmap()
   const bonds=this.initial.bonds.slice()
   for(let i=0;i<this.bondCount;i++)bonds.set(states.subarray(i*4,i*4+4),i*8+4)
   this.ensure()
   return {positions,velocities,endpoints:this.initial.endpoints,bonds}
  }finally{for(const b of staged)b.destroy()}
 }
 dispose(){if(this.disposed)return;this.disposed=true;this.gpu.dispose()}
}

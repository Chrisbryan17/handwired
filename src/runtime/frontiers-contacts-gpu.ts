import {SpatialGridGpu} from './frontiers-spatial-gpu.ts'
import {validateContactSettings,validateSphereBodies,
  type SphereBodies,type SphereContactSettings} from './frontiers-contacts.ts'
import {type SpatialGridConfig} from './frontiers-spatial.ts'

/** WebGPU fixed-radius sphere contact solver built directly on the already
 * independently validated GPU spatial index. This is real computational
 * code, but requires its own hardware validation before any GPU physics claim.
 *
 * Single-Jacobi-iteration restitution, kinetic friction and penetration
 * correction on predicted contact positions. No fracture, articulated joints
 * or CCD, and no learned model. All recurrent buffers remain GPU resident.
 */
export const sphereGpuContactWgsl=`
struct Parameters {
  origin:vec4<f32>,
  dimensions:vec4<u32>,
  motion:vec4<f32>,
  materials:vec4<f32>,
  bounds:vec4<f32>,
  count:u32,pad0:u32,pad1:u32,pad2:u32,
};
@group(0) @binding(0) var<storage,read> sourcePositions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> sourceVelocities:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> predictedPositions:array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> predictedVelocities:array<vec4<f32>>;
@group(0) @binding(4) var<storage,read_write> cellHeads:array<atomic<i32>>;
@group(0) @binding(5) var<storage,read_write> nextInCell:array<i32>;
@group(0) @binding(6) var<uniform> parameters:Parameters;
@group(0) @binding(7) var<storage,read_write> resolvedPositions:array<vec4<f32>>;
@group(0) @binding(8) var<storage,read_write> resolvedVelocities:array<vec4<f32>>;

fn coords(p:vec3<f32>)->vec3<i32>{
  return vec3<i32>(floor((p-parameters.origin.xyz)/parameters.bounds.x));
}
fn inside(c:vec3<i32>)->bool{
  return all(c>=vec3<i32>(0)) &&
    c.x<i32(parameters.dimensions.x) &&
    c.y<i32(parameters.dimensions.y) &&
    c.z<i32(parameters.dimensions.z);
}
fn cellKey(c:vec3<i32>)->u32{
  return (u32(c.z)*parameters.dimensions.y+u32(c.y))*parameters.dimensions.x+u32(c.x);
}

@compute @workgroup_size(128)
fn predict(@builtin(global_invocation_id) tid:vec3<u32>){
  let i=tid.x;
  if(i>=parameters.count){return;}
  let pos=sourcePositions[i];
  let vel=sourceVelocities[i];
  var nextPos=pos;
  var nextVel=vel;
  if(vel.w>0.0){
    nextVel.xyz=vel.xyz+parameters.motion.yzw*parameters.motion.x;
    nextPos.xyz=pos.xyz+nextVel.xyz*parameters.motion.x;
  }
  predictedPositions[i]=nextPos;
  predictedVelocities[i]=nextVel;
}

@compute @workgroup_size(128)
fn contacts(@builtin(global_invocation_id) tid:vec3<u32>){
  let i=tid.x;
  if(i>=parameters.count){return;}
  let a=sourcePositions[i];
  let velocity=sourceVelocities[i];
  let invA=velocity.w;
  let base=coords(a.xyz);
  var dv=vec3<f32>(0.0);
  var dp=vec3<f32>(0.0);
  if(inside(base)&&invA>0.0){
    for(var z=-1;z<=1;z=z+1){
      for(var y=-1;y<=1;y=y+1){
        for(var x=-1;x<=1;x=x+1){
          let c=base+vec3<i32>(x,y,z);
          if(!inside(c)){continue;}
          var node=atomicLoad(&cellHeads[cellKey(c)]);
          var visited=0u;
          loop{
            if(node<0||visited>=parameters.count){break;}
            let j=u32(node);
            if(j!=i){
              let b=sourcePositions[j];
              let velB=sourceVelocities[j];
              let invB=velB.w;
              let sumInv=invA+invB;
              if(sumInv>0.0){
                let delta=b.xyz-a.xyz;
                let d2=dot(delta,delta);
                let combined=a.w+b.w;
                if(d2<=combined*combined){
                  let dist=sqrt(d2);
                  var normal=vec3<f32>(0.0);
                  if(dist>1.0e-7){
                    normal=delta/dist;
                  }else{
                    normal=select(vec3<f32>(1.0,0.0,0.0),
                      vec3<f32>(-1.0,0.0,0.0),i>j);
                  }
                  let relative=velB.xyz-velocity.xyz;
                  let speed=dot(relative,normal);
                  let penetration=max(0.0,combined-dist);
                  if(penetration>0.0||speed<0.0){
                    var normalImpulse=0.0;
                    if(speed<0.0){
                      normalImpulse=-(1.0+parameters.materials.x)*speed/sumInv;
                    }
                    let tangent=relative-speed*normal;
                    let tLen=length(tangent);
                    var frictionImpulse=vec3<f32>(0.0);
                    if(tLen>1.0e-8){
                      let jt=min(tLen/sumInv,parameters.materials.y*normalImpulse);
                      frictionImpulse=-tangent*(jt/tLen);
                    }
                    let impulse=normalImpulse*normal+frictionImpulse;
                    dv=dv-invA*impulse;
                    let correction=max(0.0,penetration-parameters.materials.w)*
                      parameters.materials.z/sumInv;
                    dp=dp-normal*(correction*invA);
                  }
                }
              }
            }
            node=nextInCell[j];
            visited=visited+1u;
          }
        }
      }
    }
  }
  resolvedPositions[i]=vec4<f32>(a.xyz+dp,a.w);
  resolvedVelocities[i]=vec4<f32>(velocity.xyz+dv,invA);
}
`
const bits={STORAGE:128,COPY_SRC:4,COPY_DST:8,UNIFORM:64,MAP_READ:1}
export class SphereContactGpu {
  readonly grid:SpatialGridGpu
  readonly config:SphereContactSettings
  readonly count:number
  private readonly velocities:GPUBuffer
  private readonly predictedPositions:GPUBuffer
  private readonly predictedVelocities:GPUBuffer
  private readonly resolvedPositions:GPUBuffer
  private readonly resolvedVelocities:GPUBuffer
  private readonly parameters:GPUBuffer
  private readonly prediction:GPUComputePipeline
  private readonly response:GPUComputePipeline
  private readonly binding:GPUBindGroup
  private disposed=false
  private ticks=0
  private constructor(grid:SpatialGridGpu,config:SphereContactSettings,
    buffers:readonly [GPUBuffer,GPUBuffer,GPUBuffer,GPUBuffer,GPUBuffer,GPUBuffer],
    pipelines:readonly [GPUComputePipeline,GPUComputePipeline],binding:GPUBindGroup){
    this.grid=grid;this.config=config;this.count=grid.pointCount
    ;[this.velocities,this.predictedPositions,this.predictedVelocities,
      this.resolvedPositions,this.resolvedVelocities,this.parameters]=buffers
    ;[this.prediction,this.response]=pipelines
    this.binding=binding
  }
  static async create(state:SphereBodies,spatial:SpatialGridConfig,
    settings:SphereContactSettings):Promise<SphereContactGpu>{
    const count=validateSphereBodies(state)
    validateContactSettings(settings)
    if(settings.iterations!==1)
      throw new Error('GPU Jacobi solver currently admits exactly one contact iteration')
    let maximumRadius=0
    for(let i=0;i<count;i++)maximumRadius=Math.max(maximumRadius,state.positions[i*4+3]!)
    if(maximumRadius*2>spatial.radius ||
      spatial.radius>spatial.cellSize)
      throw new Error('GPU grid radius must enclose every possible sphere contact')
    const grid=await SpatialGridGpu.create(state.positions,spatial)
    try{
      const device=grid.gpu.device
      const nBytes=count*16
      const alloc=(len:number,usage:number,label:string)=>
        grid.gpu.allocate(len,usage,label)
      const velocities=alloc(nBytes,bits.STORAGE|bits.COPY_DST|bits.COPY_SRC,'Sphere velocities & inverse mass')
      const predictedPositions=alloc(nBytes,bits.STORAGE|bits.COPY_SRC,'Predicted sphere centers')
      const predictedVelocities=alloc(nBytes,bits.STORAGE|bits.COPY_SRC,'Predicted sphere velocities')
      const resolvedPositions=alloc(nBytes,bits.STORAGE|bits.COPY_SRC,'Sphere contact-corrected positions')
      const resolvedVelocities=alloc(nBytes,bits.STORAGE|bits.COPY_SRC,'Sphere contact-corrected velocities')
      const parameters=alloc(96,bits.UNIFORM|bits.COPY_DST,'Sphere contact material/step settings')
      device.queue.writeBuffer(velocities,0,state.velocities)
      const block=new ArrayBuffer(96),view=new DataView(block)
      for(let i=0;i<3;i++)view.setFloat32(i*4,spatial.origin[i]!,true)
      for(let i=0;i<3;i++)view.setUint32(16+i*4,spatial.dimensions[i]!,true)
      const motion=[settings.dt,...settings.gravity]
      const material=[settings.restitution,settings.friction,
        settings.correctionFraction,settings.penetrationSlop]
      for(let i=0;i<4;i++){
        view.setFloat32(32+i*4,motion[i]!,true)
        view.setFloat32(48+i*4,material[i]!,true)
      }
      view.setFloat32(64,spatial.cellSize,true)
      view.setUint32(80,count,true)
      device.queue.writeBuffer(parameters,0,block)
      const module=device.createShaderModule({code:sphereGpuContactWgsl,label:'Rigid sphere GPU contact and prediction'})
      const issues=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error')
      if(issues.length)throw new Error('Sphere contacts WGSL: '+issues.map(m=>m.message).join('; '))
      const layout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:4,buffer:{type:'read-only-storage'}},
        {binding:1,visibility:4,buffer:{type:'storage'}},
        {binding:2,visibility:4,buffer:{type:'storage'}},
        {binding:3,visibility:4,buffer:{type:'storage'}},
        {binding:4,visibility:4,buffer:{type:'storage'}},
        {binding:5,visibility:4,buffer:{type:'storage'}},
        {binding:6,visibility:4,buffer:{type:'uniform'}},
        {binding:7,visibility:4,buffer:{type:'storage'}},
        {binding:8,visibility:4,buffer:{type:'storage'}},
      ]})
      const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]})
      const pipelines=await Promise.all(['predict','contacts'].map(entryPoint=>
        device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})))
      const binding=device.createBindGroup({layout,entries:[
        {binding:0,resource:{buffer:grid.positions}},
        {binding:1,resource:{buffer:velocities}},
        {binding:2,resource:{buffer:predictedPositions}},
        {binding:3,resource:{buffer:predictedVelocities}},
        {binding:4,resource:{buffer:grid.cellHeads}},
        {binding:5,resource:{buffer:grid.next}},
        {binding:6,resource:{buffer:parameters}},
        {binding:7,resource:{buffer:resolvedPositions}},
        {binding:8,resource:{buffer:resolvedVelocities}},
      ]})
      return new SphereContactGpu(grid,settings,
        [velocities,predictedPositions,predictedVelocities,resolvedPositions,
          resolvedVelocities,parameters],
        [pipelines[0]!,pipelines[1]!],binding)
    }catch(error){grid.dispose();throw error}
  }
  get steps(){return this.ticks}
  private ensure(){
    if(this.disposed)throw new Error('GPU sphere-contact solver disposed')
    this.grid.gpu.check()
  }
  encodeStep(encoder:GPUCommandEncoder){
    this.ensure()
    const device=this.grid.gpu.device
    const bytes=this.count*16
    let pass=encoder.beginComputePass({label:'Contact solver: semi-implicit GPU sphere prediction'})
    pass.setPipeline(this.prediction);pass.setBindGroup(0,this.binding)
    pass.dispatchWorkgroups(Math.ceil(this.count/128));pass.end()
    encoder.copyBufferToBuffer(this.predictedPositions,0,this.grid.positions,0,bytes)
    encoder.copyBufferToBuffer(this.predictedVelocities,0,this.velocities,0,bytes)
    // GPU-resident spatial reset/scatter/query, no CPU synchronization.
    this.grid.encodeQuery(encoder)
    pass=encoder.beginComputePass({label:'Contact solver: Jacobi GPU neighbor impulses'})
    pass.setPipeline(this.response);pass.setBindGroup(0,this.binding)
    pass.dispatchWorkgroups(Math.ceil(this.count/128));pass.end()
    encoder.copyBufferToBuffer(this.resolvedPositions,0,this.grid.positions,0,bytes)
    encoder.copyBufferToBuffer(this.resolvedVelocities,0,this.velocities,0,bytes)
    this.ticks++
    void device
  }
  step(){
    this.ensure()
    const command=this.grid.gpu.device.createCommandEncoder()
    this.encodeStep(command)
    this.grid.gpu.device.queue.submit([command.finish()])
  }
  async readState():Promise<SphereBodies>{
    this.ensure()
    const device=this.grid.gpu.device,bytes=this.count*16
    const pStage=device.createBuffer({size:bytes,usage:bits.COPY_DST|bits.MAP_READ})
    const vStage=device.createBuffer({size:bytes,usage:bits.COPY_DST|bits.MAP_READ})
    try{
      // The spatial position buffer is GPU only STORAGE|COPY_DST and deliberately
      // lacks COPY_SRC. Read the last corrected-position output instead.
      const encoder=device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.resolvedPositions,0,pStage,0,bytes)
      encoder.copyBufferToBuffer(this.velocities,0,vStage,0,bytes)
      device.queue.submit([encoder.finish()])
      await Promise.all([pStage.mapAsync(1),vStage.mapAsync(1)])
      const state={positions:new Float32Array(pStage.getMappedRange().slice(0)),
        velocities:new Float32Array(vStage.getMappedRange().slice(0))}
      pStage.unmap();vStage.unmap()
      this.ensure()
      return state
    }finally{pStage.destroy();vStage.destroy()}
  }
  async flush(){this.ensure();await this.grid.flush()}
  dispose(){if(this.disposed)return;this.disposed=true;this.grid.dispose()}
}

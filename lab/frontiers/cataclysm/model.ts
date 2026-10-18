import {Circuit} from '../../../src/runtime/numeric-ui-core.ts'

/** Independent 3D weakly-compressible SPH reference with compact kernels.
 * A fluid dynamics algorithm, not animated waves, but not yet a calibrated
 * coastal tsunami model, incompressible FLIP, or GPU hardware achievement.
 */
export type Vector3=readonly [number,number,number]
export type SphState={
 positions:Float32Array // vec4 [x,y,z,mass]
 velocities:Float32Array // vec4 [vx,vy,vz,unused]
}
export type SphSettings={
 dt:number,h:number,restDensity:number,stiffness:number,
 viscosity:number,gravity:Vector3,
 min:Vector3,max:Vector3,boundaryRestitution:number,
}
export const sphDefaults:SphSettings={
 dt:.004,h:1.25,restDensity:3.5,stiffness:8,viscosity:.02,
 gravity:[0,-9.81,0],min:[-7,-3,-6],max:[7,8,6],
 boundaryRestitution:.2,
}
export function validateSph(state:SphState,settings:SphSettings=sphDefaults,
 maxParticles=1024){
 if(state.positions.length===0||state.positions.length%4!==0||
  state.positions.length!==state.velocities.length||
  state.positions.length/4>maxParticles||
  !Number.isSafeInteger(maxParticles)||maxParticles<1||maxParticles>131_072)
  throw new Error('invalid SPH particle arrays or reference budget')
 if(!Number.isFinite(settings.dt)||settings.dt<=0||settings.dt>.02||
  !Number.isFinite(settings.h)||settings.h<.05||settings.h>20||
  !Number.isFinite(settings.restDensity)||settings.restDensity<=0||
  !Number.isFinite(settings.stiffness)||settings.stiffness<0||settings.stiffness>1e5||
  !Number.isFinite(settings.viscosity)||settings.viscosity<0||settings.viscosity>5||
  !Number.isFinite(settings.boundaryRestitution)||
  settings.boundaryRestitution<0||settings.boundaryRestitution>1||
  !settings.gravity.every(v=>Number.isFinite(v)&&Math.abs(v)<=200))
  throw new Error('invalid SPH simulation parameters')
 for(let axis=0;axis<3;axis++)
  if(!Number.isFinite(settings.min[axis]!)||
   !Number.isFinite(settings.max[axis]!)||
   settings.max[axis]!-settings.min[axis]!<=.02)
   throw new Error('invalid fluid boundary')
 for(let i=0;i<state.positions.length;i++){
  if(!Number.isFinite(state.positions[i]!)||!Number.isFinite(state.velocities[i]!))
   throw new Error('nonfinite fluid state')
  if(i%4===3&&state.positions[i]!<=0)throw new Error('fluid particle mass must be positive')
  if(i%4<3&&(state.positions[i]!<settings.min[i%4]!||
   state.positions[i]!>settings.max[i%4]!))
   throw new Error('particle initially outside fluid domain')
 }
 return state.positions.length/4
}
export function poly6(distanceSquared:number,h:number):number{
 if(!Number.isFinite(h)||h<=0||!Number.isFinite(distanceSquared)||distanceSquared<0)
  throw new Error('invalid compact density kernel')
 if(distanceSquared>=h*h)return 0
 return (315/(64*Math.PI*h**9))*(h*h-distanceSquared)**3
}
export function spikyRadialDerivative(r:number,h:number):number{
 if(!Number.isFinite(r)||r<0||!Number.isFinite(h)||h<=0)
  throw new Error('invalid spiky pressure kernel')
 return r>=h?0:-45/(Math.PI*h**6)*(h-r)**2
}
export function viscosityLaplacian(r:number,h:number):number{
 if(!Number.isFinite(r)||r<0||!Number.isFinite(h)||h<=0)
  throw new Error('invalid viscosity kernel')
 return r>=h?0:45/(Math.PI*h**6)*(h-r)
}
export function buildSphPressureCircuit(){
 const c=new Circuit(3)
 const density=c.source(0),stiffness=c.source(1),rest=c.source(2)
 const delta=c.roundAffine([[density,1],[rest,-1]])
 const signedPressure=c.multiply(stiffness,delta)
 return {circuit:c,outputWires:[signedPressure]}
}
export function densityPressureCpu(state:SphState,p:SphSettings=sphDefaults){
 const n=validateSph(state,p)
 const density=new Float32Array(n),pressure=new Float32Array(n)
 const h2=p.h*p.h
 for(let i=0;i<n;i++){
  const a=i*4
  let sum=0
  for(let j=0;j<n;j++){
   const b=j*4
   const dx=state.positions[a]!-state.positions[b]!
   const dy=state.positions[a+1]!-state.positions[b+1]!
   const dz=state.positions[a+2]!-state.positions[b+2]!
   const d2=dx*dx+dy*dy+dz*dz
   if(d2<h2)sum+=state.positions[b+3]!*poly6(d2,p.h)
  }
  density[i]=Math.fround(sum)
  pressure[i]=Math.fround(Math.max(0,p.stiffness*(sum-p.restDensity)))
 }
 return {density,pressure}
}
export type SphDiagnostics={mass:number,centerOfMass:Vector3,
  kinetic:number,minDensity:number,maxDensity:number,
  maxSpeed:number,nonfinite:number}
export function sphDiagnostics(state:SphState,p:SphSettings=sphDefaults):SphDiagnostics{
 const n=validateSph(state,p)
 const densities=densityPressureCpu(state,p).density
 let mass=0,kinetic=0,maxSpeed=0,nonfinite=0
 const weighted:[number,number,number]=[0,0,0]
 let minDensity=Infinity,maxDensity=0
 for(let i=0;i<n;i++){
  const at=i*4,m=state.positions[at+3]!
  mass+=m
  let speed2=0
  for(let k=0;k<3;k++){
   weighted[k]=weighted[k]!+m*state.positions[at+k]!
   speed2+=state.velocities[at+k]!**2
  }
  const speed=Math.sqrt(speed2)
  maxSpeed=Math.max(maxSpeed,speed)
  kinetic+=.5*m*speed2
  minDensity=Math.min(minDensity,densities[i]!)
  maxDensity=Math.max(maxDensity,densities[i]!)
  if(!Number.isFinite(speed))nonfinite++
 }
 return {mass,centerOfMass:weighted.map(x=>x/mass) as unknown as Vector3,
  kinetic,minDensity,maxDensity,maxSpeed,nonfinite}
}
export function stepSphCpu(initial:SphState,p:SphSettings=sphDefaults):SphState{
 const n=validateSph(initial,p)
 const field=densityPressureCpu(initial,p)
 const after:SphState={positions:initial.positions.slice(),velocities:initial.velocities.slice()}
 for(let i=0;i<n;i++){
  const at=i*4
  const x=initial.positions[at]!,y=initial.positions[at+1]!,z=initial.positions[at+2]!
  const rhoI=field.density[i]!,presI=field.pressure[i]!
  const acc:[number,number,number]=[...p.gravity]
  for(let j=0;j<n;j++){
   if(i===j)continue
   const jt=j*4
   const d:[number,number,number]=[
    x-initial.positions[jt]!,y-initial.positions[jt+1]!,z-initial.positions[jt+2]!]
   const r=Math.hypot(...d)
   if(r>=p.h||r<1e-8)continue
   const rhoJ=field.density[j]!,presJ=field.pressure[j]!
   const massJ=initial.positions[jt+3]!
   const coefficient=-massJ*(presI/(rhoI*rhoI)+presJ/(rhoJ*rhoJ))*
     spikyRadialDerivative(r,p.h)/r
   const lap=viscosityLaplacian(r,p.h)
   for(let k=0;k<3;k++)acc[k]=acc[k]!+coefficient*d[k]!+
     p.viscosity*massJ*(initial.velocities[jt+k]!-initial.velocities[at+k]!)*lap/rhoJ
  }
  for(let k=0;k<3;k++){
   const velocity=Math.fround(initial.velocities[at+k]!+p.dt*acc[k]!)
   const moved=Math.fround(initial.positions[at+k]!+p.dt*velocity)
   if(moved<p.min[k]!){
    after.positions[at+k]=Math.fround(p.min[k]!)
    after.velocities[at+k]=Math.fround(-velocity*p.boundaryRestitution)
   }else if(moved>p.max[k]!){
    after.positions[at+k]=Math.fround(p.max[k]!)
    after.velocities[at+k]=Math.fround(-velocity*p.boundaryRestitution)
   }else{
    after.positions[at+k]=moved
    after.velocities[at+k]=velocity
   }
  }
 }
 return after
}
export function makeFluidBlock(nx=7,ny=6,nz=6,spacing=.57):SphState{
 if(![nx,ny,nz].every(v=>Number.isSafeInteger(v)&&v>=2&&v<=12)||
  !Number.isFinite(spacing)||spacing<.1||spacing>1)
  throw new Error('invalid SPH source block')
 const positions:number[]=[],velocities:number[]=[]
 for(let i=0;i<nx;i++)for(let j=0;j<ny;j++)for(let k=0;k<nz;k++){
  positions.push(Math.fround(-4.6+i*spacing),Math.fround(-2+j*spacing),
   Math.fround(-2+k*spacing),1)
  velocities.push(0,0,0,0)
 }
 const state={positions:Float32Array.from(positions),
   velocities:Float32Array.from(velocities)}
 validateSph(state)
 return state
}

import {makeFluidBlock,validateSph,sphDefaults,
 type SphSettings,type SphState} from './model.ts'
export type CataclysmPreset='dam-break'|'counter-current'|'droplet'|'dense-4096'|'stress-18432'
/** Repeatable initial conditions for the real WCSPH numerical solver.
 * Particle motion is never sampled from baked wave animations.
 */
export function makeCataclysmPreset(name:CataclysmPreset):SphState{
 if(!(['dam-break','counter-current','droplet','dense-4096','stress-18432'] as string[]).includes(name))
  throw new Error('unrecognized real fluid scenario')
 if(name==='stress-18432')return makeGpuStressField()
 if(name==='dense-4096')return makeGpuStressField(16,16,16)
 if(name==='dam-break'){
  const state=makeFluidBlock(10,8,8,.48)
  for(let i=0;i<state.positions.length/4;i++){
   const at=i*4
   state.velocities[at]=Math.fround(.6+state.positions[at+1]!*-.04)
  }
  return state
 }
 if(name==='counter-current'){
  const original=makeFluidBlock(8,7,7,.47)
  const n=original.positions.length/4
  const positions=new Float32Array(n*8),velocities=new Float32Array(n*8)
  for(let i=0;i<n;i++){
   const base=i*4,mirror=(n+i)*4
   positions.set(original.positions.subarray(base,base+4),base)
   positions.set(original.positions.subarray(base,base+4),mirror)
   positions[mirror]=Math.fround(positions[mirror]!+5.3)
   velocities[base]=2.5
   velocities[mirror]=-2.5
  }
  return {positions,velocities}
 }
 {
  const state=makeFluidBlock(10,8,7,.48)
  for(let i=0;i<state.positions.length/4;i++){
   const at=i*4
   state.positions[at]=Math.fround(state.positions[at]!+1.8)
   state.positions[at+1]=Math.fround(state.positions[at+1]!+3.5)
   state.positions[at+2]=Math.fround(state.positions[at+2]!+.4)
   state.velocities[at]=Math.fround(-.3*state.positions[at+2]!)
   state.velocities[at+2]=Math.fround(.3*state.positions[at])
  }
  return state
 }
}
export function cataclysmSettings(gravity:number,stiffness:number,viscosity:number):SphSettings{
 const p:SphSettings={...sphDefaults,gravity:[0,-gravity,0],
  stiffness,restDensity:2.8,viscosity}
 if(!Number.isFinite(gravity)||gravity<0||gravity>25||
  !Number.isFinite(stiffness)||stiffness<1||stiffness>60||
  !Number.isFinite(viscosity)||viscosity<0||viscosity>1)
  throw new Error('invalid fluid user controls')
 return p
}
export function inspectPreset(name:CataclysmPreset,p:SphSettings=cataclysmSettings(9.81,12,.12)){
 const state=makeCataclysmPreset(name)
 const n=validateSph(state,p,state.positions.length>4096?131_072:1024)
 let totalMass=0,activeMomentum=0
 for(let i=0;i<n;i++){
  totalMass+=state.positions[i*4+3]!
  activeMomentum+=Math.abs(state.velocities[i*4]!)+Math.abs(state.velocities[i*4+1]!)
 }
 return {n,totalMass,activeMomentum}
}

/** Large, deterministic real WCSPH state for *hardware-only* throughput and
 * stability testing. CPU all-pairs oracle deliberately remains capped at
 * 1024 points; large GPU scenes only carry conservation/invariant evidence.
 */
export function makeGpuStressField(nx=32,ny=24,nz=24,spacing=.34):SphState{
 const n=nx*ny*nz
 if(![nx,ny,nz].every(x=>Number.isSafeInteger(x)&&x>=2&&x<=72)||
  !Number.isFinite(spacing)||spacing<.25||spacing>.65||
  !Number.isSafeInteger(n)||n>131_072)
  throw new Error('invalid large GPU SPH workload')
 const positions=new Float32Array(n*4)
 const velocities=new Float32Array(n*4)
 let index=0
 for(let y=0;y<ny;y++)for(let z=0;z<nz;z++)for(let x=0;x<nx;x++){
  const at=index*4
  positions[at]=Math.fround(-4.6+x*spacing)
  positions[at+1]=Math.fround(-2+y*spacing)
  positions[at+2]=Math.fround(-2+z*spacing)
  positions[at+3]=1
  velocities[at]=Math.fround(.15+((index*11)%13-6)*.018)
  velocities[at+2]=Math.fround(((index*5)%17-8)*.017)
  index++
 }
 const state={positions,velocities}
 validateSph(state,{
  ...sphDefaults,dt:.001,restDensity:25,stiffness:2.5,viscosity:.025},131_072)
 return state
}

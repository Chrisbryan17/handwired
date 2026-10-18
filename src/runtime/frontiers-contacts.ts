import {Circuit} from './numeric-ui-core.ts'

/**
 * Shared independent CPU narrow-phase for a sphere-contact dynamics slice.
 *
 * Equal-and-opposite normal and Coulomb kinetic friction impulses, with
 * inverse-mass-weighted positional correction. Integrates gravity and drift.
 * This is **not** a continuum deformation/structural fracture solver, and
 * the GPU spatial grid does not automatically make this CPU solver GPU-bound.
 */
export type Three = readonly [number,number,number]
export type SphereBodies={
  positions:Float32Array // x,y,z,radius
  velocities:Float32Array // vx,vy,vz,inverseMass; inverseMass 0 => anchored
}
export type SphereContactSettings={
  dt:number
  gravity:Three
  restitution:number
  friction:number
  penetrationSlop:number
  correctionFraction:number
  iterations:number
}
export type ContactResult={
  state:SphereBodies
  contactsProcessed:number
  peakPenetration:number
  iterationsUsed:number
}
export const sphereContactDefaults:SphereContactSettings={
  dt:1/60,gravity:[0,-9.81,0],
  restitution:.2,friction:.45,
  penetrationSlop:.001,correctionFraction:.75,iterations:4,
}
export function validateSphereBodies(state:SphereBodies):number{
  if(state.positions.length===0||state.positions.length%4!==0||
    state.positions.length!==state.velocities.length||
    state.positions.length/4>10000)throw new Error('invalid sphere state shape')
  for(let i=0;i<state.positions.length;i++){
    const p=state.positions[i]!,v=state.velocities[i]!
    if(!Number.isFinite(p)||!Number.isFinite(v))
      throw new Error('nonfinite rigid-sphere state')
    if(i%4===3&&(p<=0||v<0))
      throw new Error('radius must be positive and inverse mass nonnegative')
  }
  return state.positions.length/4
}
export function validateContactSettings(p:SphereContactSettings):void{
  if(!Number.isFinite(p.dt)||p.dt<0||p.dt>.1||
    !p.gravity.every(x=>Number.isFinite(x)&&Math.abs(x)<=200)||
    !Number.isFinite(p.restitution)||p.restitution<0||p.restitution>1||
    !Number.isFinite(p.friction)||p.friction<0||p.friction>5||
    !Number.isFinite(p.penetrationSlop)||p.penetrationSlop<0||p.penetrationSlop>.5||
    !Number.isFinite(p.correctionFraction)||p.correctionFraction<0||p.correctionFraction>1||
    !Number.isSafeInteger(p.iterations)||p.iterations<1||p.iterations>20)
    throw new Error('invalid rigid sphere contact parameters')
}
export function buildContactImpulseCircuit(){
  const circuit=new Circuit(4)
  const [velocity,impulse,normal,inverseMass]=[0,1,2,3].map(i=>circuit.source(i))
  const directed=circuit.multiply(impulse!,normal!)
  const weighted=circuit.multiply(directed,inverseMass!)
  const updated=circuit.roundAffine([[velocity!,1],[weighted,1]])
  return {circuit,outputWires:[updated]}
}
export function rigidBodyDiagnostics(state:SphereBodies) {
  const count=validateSphereBodies(state)
  let kinetic=0,moving=0
  const momentum:[number,number,number]=[0,0,0]
  for(let i=0;i<count;i++){
    const base=i*4,inv=state.velocities[base+3]!
    if(inv<=0)continue
    moving++
    const mass=1/inv
    let speed2=0
    for(let k=0;k<3;k++){
      const velocity=state.velocities[base+k]!
      momentum[k]+=mass*velocity
      speed2+=velocity*velocity
    }
    kinetic+=.5*mass*speed2
  }
  return {count,moving,momentum,kinetic}
}

/** Brute-force O(N²) independent reference, one state advance.
 * Contact pair impulses accumulate Jacobi-style within each solver iteration,
 * so no GPU atomic update ordering is implicitly assumed.
 */
export function stepSphereContactsCpu(initial:SphereBodies,
  settings:SphereContactSettings=sphereContactDefaults):ContactResult{
  const n=validateSphereBodies(initial)
  validateContactSettings(settings)
  const p=initial.positions.slice()
  const v=initial.velocities.slice()
  // Semi-implicit Euler predictor. Anchored bodies cannot drift.
  for(let i=0;i<n;i++){
    const at=i*4,inv=v[at+3]!
    if(inv<=0)continue
    for(let axis=0;axis<3;axis++){
      v[at+axis]=Math.fround(v[at+axis]!+settings.gravity[axis]!*settings.dt)
      p[at+axis]=Math.fround(p[at+axis]!+v[at+axis]!*settings.dt)
    }
  }
  let contactsProcessed=0,peakPenetration=0,iterationsUsed=0
  for(let iteration=0;iteration<settings.iterations;iteration++){
    const deltaVelocity=new Float64Array(n*3)
    const deltaPosition=new Float64Array(n*3)
    let contactsThisIteration=0
    for(let i=0;i<n;i++)for(let j=i+1;j<n;j++){
      const a=i*4,b=j*4
      const invA=v[a+3]!,invB=v[b+3]!,sumInv=invA+invB
      if(sumInv<=0)continue
      const dx=p[b]!-p[a]!,dy=p[b+1]!-p[a+1]!,dz=p[b+2]!-p[a+2]!
      const maxDistance=p[a+3]!+p[b+3]!
      const distSq=dx*dx+dy*dy+dz*dz
      if(distSq>maxDistance*maxDistance)continue
      const distance=Math.sqrt(distSq)
      const nx=distance>1e-12?dx/distance:1
      const ny=distance>1e-12?dy/distance:0
      const nz=distance>1e-12?dz/distance:0
      const depth=Math.max(0,maxDistance-distance)
      const rx=v[b]!-v[a]!,ry=v[b+1]!-v[a+1]!,rz=v[b+2]!-v[a+2]!
      const speed=rx*nx+ry*ny+rz*nz
      if(depth<=0&&speed>=0)continue
      contactsProcessed++;contactsThisIteration++
      peakPenetration=Math.max(peakPenetration,depth)
      const normalImpulse=speed<0?-(1+settings.restitution)*speed/sumInv:0
      const tangX=rx-speed*nx,tangY=ry-speed*ny,tangZ=rz-speed*nz
      const tangMagnitude=Math.hypot(tangX,tangY,tangZ)
      const jt=tangMagnitude>1e-12 ?
        Math.min(tangMagnitude/sumInv,settings.friction*normalImpulse):0
      const tangentScale=tangMagnitude>1e-12 ? -jt/tangMagnitude:0
      const ix=normalImpulse*nx+tangentScale*tangX
      const iy=normalImpulse*ny+tangentScale*tangY
      const iz=normalImpulse*nz+tangentScale*tangZ
      const correction=Math.max(0,depth-settings.penetrationSlop)*
        settings.correctionFraction/sumInv
      const va=i*3,vb=j*3
      deltaVelocity[va]-=ix*invA;deltaVelocity[va+1]-=iy*invA
      deltaVelocity[va+2]-=iz*invA
      deltaVelocity[vb]+=ix*invB;deltaVelocity[vb+1]+=iy*invB
      deltaVelocity[vb+2]+=iz*invB
      deltaPosition[va]-=correction*nx*invA
      deltaPosition[va+1]-=correction*ny*invA
      deltaPosition[va+2]-=correction*nz*invA
      deltaPosition[vb]+=correction*nx*invB
      deltaPosition[vb+1]+=correction*ny*invB
      deltaPosition[vb+2]+=correction*nz*invB
    }
    if(contactsThisIteration===0)break
    iterationsUsed++
    for(let i=0;i<n;i++){
      const at=i*4,base=i*3
      for(let axis=0;axis<3;axis++){
        v[at+axis]=Math.fround(v[at+axis]!+deltaVelocity[base+axis]!)
        p[at+axis]=Math.fround(p[at+axis]!+deltaPosition[base+axis]!)
      }
    }
  }
  return {state:{positions:p,velocities:v},contactsProcessed,
    peakPenetration,iterationsUsed}
}

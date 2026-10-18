import {Circuit} from '../../../src/runtime/numeric-ui-core.ts'

/** Normalized Newtonian point-mass model with Plummer softening.
 * Not cosmological galaxy formation and not an unsoftened direct N-body solver.
 * Every simulated body contributes to and responds to gravity (O(N²)).
 */
export const FIXED_DT = 0.003
export const DEFAULT_G = 0.55
export const DEFAULT_SOFTENING = 0.045
export const STRIDE = 8 // position xyz + mass; velocity xyz + visual kind
export type UniverseState = Float32Array
export type UniverseParams = {gravity:number,softening:number}
export const defaultParams: UniverseParams = {gravity:DEFAULT_G,softening:DEFAULT_SOFTENING}

export function buildUniverseCircuit() {
  const circuit = new Circuit(9)
  const s = Array.from({length:9},(_,i)=>circuit.source(i))
  const velocities = [0,1,2].map(i => circuit.roundAffine([
    [s[3+i]!,1],[s[6+i]!,FIXED_DT],
  ]))
  const positions = [0,1,2].map(i=>circuit.roundAffine([
    [s[i]!,1],[velocities[i]!,FIXED_DT],
  ]))
  return {circuit,outputWires:[...positions,...velocities] as const}
}
function random(seed:number) {
  let value=seed>>>0
  return () => {
    value=(Math.imul(value,1664525)+1013904223)>>>0
    return value/4294967296
  }
}

/** Two tilted stellar disks, including two massive central bodies.
 * Positions are initial conditions, never interpreted as observational data.
 */
export function makeCollision(count=256,seed=37):UniverseState {
  if(!Number.isSafeInteger(count)||count<4||count>16384||!Number.isSafeInteger(seed))
    throw new Error('invalid N-body count or seed')
  const random01=random(seed)
  const data=new Float32Array(count*STRIDE)
  for(let i=0;i<count;i++){
    const g=i%2,centerX=g===0?-.54:.54,centerY=g===0?-.06:.06
    const core=i<2
    const radius=core?0:.065+Math.sqrt(random01())*.37
    const angle=random01()*Math.PI*2
    const tilt=g===0?.28:-.34
    const dx=Math.cos(angle)*radius,dy=Math.sin(angle)*radius
    const z=dy*Math.sin(tilt)+(random01()-.5)*.018
    const y=dy*Math.cos(tilt)
    const mass=core?.23:.0023
    const orbitalSpeed=core?0:Math.sqrt(DEFAULT_G*.36*radius*radius/
      Math.pow(radius*radius+.07*.07,1.5))
    const spin=g===0?1:-1
    const vTangentialX=-Math.sin(angle)*orbitalSpeed*spin
    const vTangentialY=Math.cos(angle)*Math.cos(tilt)*orbitalSpeed*spin
    const vTangentialZ=Math.cos(angle)*Math.sin(tilt)*orbitalSpeed*spin
    const at=i*STRIDE
    data[at]=centerX+dx;data[at+1]=centerY+y;data[at+2]=z;data[at+3]=mass
    data[at+4]=(g===0?.12:-.12)+vTangentialX
    data[at+5]=(g===0?.024:-.024)+vTangentialY
    data[at+6]=vTangentialZ;data[at+7]=core?2:g
  }
  return data
}
export function checkUniverseState(state: UniverseState, params: UniverseParams):number {
  if(state.length%STRIDE!==0||state.length<STRIDE*2)
    throw new Error('bad N-body state length')
  if(!Number.isFinite(params.gravity)||params.gravity<0||
     !Number.isFinite(params.softening)||params.softening<=0)
    throw new Error('invalid gravity/softening')
  for(let i=0;i<state.length;i++){
    if(!Number.isFinite(state[i]))throw new Error('nonfinite N-body state')
    if(i%STRIDE===3 && state[i]!<=0)throw new Error('body mass must be positive')
  }
  return state.length/STRIDE
}
/** Independently handwritten direct-force CPU oracle.
 * Snapshot forces, then symplectic Euler velocity/position updates.
 */
export function stepUniverseCpu(state:UniverseState,
  params:UniverseParams=defaultParams):UniverseState {
  const n=checkUniverseState(state,params),next=state.slice()
  const f=Math.fround
  const eps=params.softening*params.softening
  for(let i=0;i<n;i++){
    const at=i*STRIDE
    let ax=0,ay=0,az=0
    for(let j=0;j<n;j++){
      if(i===j)continue
      const jt=j*STRIDE
      const dx=state[jt]!-state[at]!,dy=state[jt+1]!-state[at+1]!,
        dz=state[jt+2]!-state[at+2]!
      const dist2=dx*dx+dy*dy+dz*dz+eps
      const weight=params.gravity*state[jt+3]!/(dist2*Math.sqrt(dist2))
      ax+=weight*dx;ay+=weight*dy;az+=weight*dz
    }
    const acceleration=[ax,ay,az]
    for(let k=0;k<3;k++){
      const v=f(state[at+4+k]!+f(f(acceleration[k]!)*f(FIXED_DT)))
      next[at+4+k]=v
      next[at+k]=f(state[at+k]!+f(v*f(FIXED_DT)))
    }
  }
  return next
}
export function universeDiagnostics(state:UniverseState,
  params:UniverseParams=defaultParams) {
  const count=checkUniverseState(state,params),momentum=[0,0,0]
  let kinetic=0,potential=0,mass=0
  for(let i=0;i<count;i++){
    const a=i*STRIDE,mi=state[a+3]!
    mass+=mi
    for(let k=0;k<3;k++)momentum[k]=momentum[k]!+mi*state[a+4+k]!
    kinetic+=.5*mi*(state[a+4]!**2+state[a+5]!**2+state[a+6]!**2)
    for(let j=i+1;j<count;j++){
      const b=j*STRIDE
      const dx=state[a]!-state[b]!,dy=state[a+1]!-state[b+1]!,dz=state[a+2]!-state[b+2]!
      potential-=params.gravity*mi*state[b+3]!/
        Math.sqrt(dx*dx+dy*dy+dz*dz+params.softening*params.softening)
    }
  }
  return {count,mass,momentum,kinetic,potential,total:kinetic+potential}
}

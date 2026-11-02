import {Circuit} from '../../../src/runtime/numeric-ui-core.ts'

/** Two revolute joints in one sagittal plane, pinned at a fixed hip.
 * Independent Euler-Lagrange point-mass dynamics with a penalty foot contact.
 * This is a reproducible articulated *leg* reference, not a free-moving
 * quadruped, humanoid, ground-truth MuJoCo or reinforced-learned gait.
 */
export type PlanarLegState={joints:Float32Array}
export type PlanarLegParameters={
 dt:number,length1:number,length2:number,mass1:number,mass2:number,
 gravity:number,hipHeight:number,groundHeight:number,
 stiffness:number,contactDamping:number,friction:number,
 tangentDamping:number,jointDamping:number,
 kp:number,kd:number,torqueLimit:number,
 targetHip:number,targetKnee:number,
}
export const legDefaults:PlanarLegParameters={
 dt:.002,length1:.75,length2:.68,mass1:1.1,mass2:.7,
 gravity:9.81,hipHeight:1.55,groundHeight:0,
 stiffness:100,contactDamping:8,friction:.7,tangentDamping:6,
 jointDamping:.14,kp:16,kd:2.5,torqueLimit:28,
 targetHip:.15,targetKnee:-.35,
}
export function validateLegParameters(p:PlanarLegParameters){
 const finite=Object.values(p).every(v=>Number.isFinite(v))
 if(!finite||p.dt<=0||p.dt>.02||
  p.length1<=0||p.length2<=0||p.length1>10||p.length2>10||
  p.mass1<=0||p.mass2<=0||p.mass1>1e4||p.mass2>1e4||
  p.gravity<0||p.gravity>200||p.hipHeight<0||p.hipHeight>100||
  p.stiffness<0||p.stiffness>100000||p.contactDamping<0||
  p.contactDamping>1000||p.friction<0||p.friction>3||
  p.tangentDamping<0||p.tangentDamping>1000||
  p.jointDamping<0||p.jointDamping>1000||
  p.kp<0||p.kp>10000||p.kd<0||p.kd>10000||
  p.torqueLimit<0||p.torqueLimit>1e6||
  Math.abs(p.targetHip)>20||Math.abs(p.targetKnee)>20)
  throw new Error('invalid planar articulated leg physical/control parameters')
}
export function validateLegPopulation(input:PlanarLegState,max=8192){
 const count=input.joints.length/4
 if(count<1||count>max||!Number.isSafeInteger(max)||max<1||
  !Number.isInteger(count)||input.joints.length%4!==0)
  throw new Error('invalid planar leg population shape/size')
 for(const v of input.joints)
  if(!Number.isFinite(v)||Math.abs(v)>100)
   throw new Error('nonfinite or dangerously out-of-range angular state')
 return count
}
/** One circuit implements both hip and knee PD motor commands. Targets and
 * gains are explicit numerical inputs; this is NOT an RL-trained policy.
 */
export function buildLegMotorCircuit(){
 const circuit=new Circuit(5)
 const q=circuit.source(0),velocity=circuit.source(1)
 const target=circuit.source(2),kp=circuit.source(3),kd=circuit.source(4)
 const error=circuit.roundAffine([[target,1],[q,-1]])
 const proportional=circuit.multiply(kp,error)
 const derivative=circuit.multiply(kd,velocity)
 const torque=circuit.roundAffine([[proportional,1],[derivative,-1]])
 return {circuit,outputWires:[torque]}
}
export function motorTorque(q:number,velocity:number,target:number,
 p:PlanarLegParameters):number{
 const raw=p.kp*(target-q)-p.kd*velocity
 return Math.max(-p.torqueLimit,Math.min(p.torqueLimit,raw))
}
export type LegSnapshot={
 knee:readonly [number,number],foot:readonly [number,number],
 footVelocity:readonly [number,number],normalForce:number,
 tangentialForce:number,kinetic:number,potential:number,
 inertia:readonly [number,number,number],determinant:number,
}
export function legSnapshot(state:readonly [number,number,number,number],
 p:PlanarLegParameters=legDefaults):LegSnapshot{
 validateLegParameters(p)
 const [q1,q2,w1,w2]=state
 if(state.some(v=>!Number.isFinite(v)||Math.abs(v)>100))
  throw new Error('invalid leg position or angular speed')
 const a=q1+q2
 const s1=Math.sin(q1),c1=Math.cos(q1)
 const sa=Math.sin(a),ca=Math.cos(a)
 const x1=p.length1*s1
 const y1=p.hipHeight-p.length1*c1
 const x2=x1+p.length2*sa
 const y2=y1-p.length2*ca
 const jx1=p.length1*c1+p.length2*ca,jx2=p.length2*ca
 const jy1=p.length1*s1+p.length2*sa,jy2=p.length2*sa
 const vx=jx1*w1+jx2*w2,vy=jy1*w1+jy2*w2
 const penetration=Math.max(0,p.groundHeight-y2)
 const normalForce=penetration>0?
  Math.max(0,p.stiffness*penetration-p.contactDamping*vy):0
 const tangentialForce=normalForce===0?0:
  Math.max(-p.friction*normalForce,Math.min(p.friction*normalForce,
   -p.tangentDamping*vx))
 const m11=(p.mass1+p.mass2)*p.length1**2+p.mass2*p.length2**2+
  2*p.mass2*p.length1*p.length2*Math.cos(q2)
 const m12=p.mass2*p.length2**2+
  p.mass2*p.length1*p.length2*Math.cos(q2)
 const m22=p.mass2*p.length2**2
 const determinant=m11*m22-m12*m12
 if(determinant<=1e-10)throw new Error('articulated mass matrix singular')
 const kinetic=.5*(m11*w1*w1+2*m12*w1*w2+m22*w2*w2)
 const potential=p.gravity*(p.mass1*y1+p.mass2*y2)
 return {knee:[x1,y1],foot:[x2,y2],footVelocity:[vx,vy],
  normalForce,tangentialForce,kinetic,potential,
  inertia:[m11,m12,m22],determinant}
}
export function stepLegCpu(input:PlanarLegState,
 p:PlanarLegParameters=legDefaults):PlanarLegState{
 const count=validateLegPopulation(input)
 validateLegParameters(p)
 const out=new Float32Array(input.joints.length)
 for(let i=0;i<count;i++){
  const at=i*4
  const q1=input.joints[at]!,q2=input.joints[at+1]!
  const w1=input.joints[at+2]!,w2=input.joints[at+3]!
  const s=legSnapshot([q1,q2,w1,w2],p)
  const [m11,m12,m22]=s.inertia
  const sinQ2=Math.sin(q2)
  const a=q1+q2
  const h=-p.mass2*p.length1*p.length2*sinQ2
  const coriolis1=h*(2*w1*w2+w2*w2)
  const coriolis2=-h*w1*w1
  const grav1=p.gravity*((p.mass1+p.mass2)*p.length1*Math.sin(q1)+
   p.mass2*p.length2*Math.sin(a))
  const grav2=p.gravity*p.mass2*p.length2*Math.sin(a)
  const jacX1=p.length1*Math.cos(q1)+p.length2*Math.cos(a)
  const jacX2=p.length2*Math.cos(a)
  const jacY1=p.length1*Math.sin(q1)+p.length2*Math.sin(a)
  const jacY2=p.length2*Math.sin(a)
  const external1=jacX1*s.tangentialForce+jacY1*s.normalForce
  const external2=jacX2*s.tangentialForce+jacY2*s.normalForce
  const rhs1=motorTorque(q1,w1,p.targetHip,p)+external1-
   coriolis1-grav1-p.jointDamping*w1
  const rhs2=motorTorque(q2,w2,p.targetKnee,p)+external2-
   coriolis2-grav2-p.jointDamping*w2
  const accel1=(m22*rhs1-m12*rhs2)/s.determinant
  const accel2=(m11*rhs2-m12*rhs1)/s.determinant
  const v1=Math.fround(w1+p.dt*accel1)
  const v2=Math.fround(w2+p.dt*accel2)
  out[at]=Math.fround(q1+p.dt*v1)
  out[at+1]=Math.fround(q2+p.dt*v2)
  out[at+2]=v1;out[at+3]=v2
 }
 return {joints:out}
}
export function makeLegPopulation(count=64,seed=44511):PlanarLegState{
 if(!Number.isSafeInteger(count)||count<1||count>8192||
  !Number.isSafeInteger(seed)||seed<0)throw new Error('bad leg population seed/count')
 const out=new Float32Array(count*4)
 let rng=seed>>>0
 const random=()=>{rng=(Math.imul(rng,1664525)+1013904223)>>>0;return rng/4294967296}
 for(let i=0;i<count;i++){
  const at=i*4
  out[at]=Math.fround((random()-.5)*.40)
  out[at+1]=Math.fround(-.28+(random()-.5)*.38)
  out[at+2]=Math.fround((random()-.5)*.45)
  out[at+3]=Math.fround((random()-.5)*.38)
 }
 return {joints:out}
}
export function legPopulationDiagnostics(input:PlanarLegState,
 p:PlanarLegParameters=legDefaults){
 const count=validateLegPopulation(input)
 let energy=0,contacts=0,penetration=0,maxSpeed=0,minDet=Infinity
 for(let i=0;i<count;i++){
  const at=i*4
  const state:[number,number,number,number]=[
   input.joints[at]!,input.joints[at+1]!,
   input.joints[at+2]!,input.joints[at+3]!]
  const s=legSnapshot(state,p)
  energy+=s.kinetic+s.potential
  if(s.normalForce>0)contacts++
  penetration=Math.max(penetration,Math.max(0,p.groundHeight-s.foot[1]))
  maxSpeed=Math.max(maxSpeed,Math.abs(state[2]),Math.abs(state[3]))
  minDet=Math.min(minDet,s.determinant)
 }
 return {count,totalEnergy:energy,contactCount:contacts,
  maximumGroundPenetration:penetration,maximumAngularSpeed:maxSpeed,
  minimumMassDeterminant:minDet}
}

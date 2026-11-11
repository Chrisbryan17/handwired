import {test,expect} from 'bun:test'
import {buildLegMotorCircuit,legDefaults,legSnapshot,motorTorque,
 makeLegPopulation,stepLegCpu,legPopulationDiagnostics,
 type PlanarLegParameters,type PlanarLegState} from './model.ts'
const zeroMotor:PlanarLegParameters={...legDefaults,
 kp:0,kd:0,gravity:0,groundHeight:-20,jointDamping:0}
test('Handwired proportional-derivative motor equals independent signed reference',()=>{
 const {circuit,outputWires}=buildLegMotorCircuit()
 for(const [q,w,target,kp,kd] of [
  [0,0,.2,20,3],[.3,-.4,-.3,12,2],[-.8,3,.1,4,8],
  [1.5,.2,-2,17,.9]
 ] as [number,number,number,number,number][]){
  const result=circuit.evaluateReference(Float32Array.of(q,w,target,kp,kd))[outputWires[0]!]!
  const expected=Math.fround(Math.fround(kp*Math.fround(target-q))-
   Math.fround(kd*w))
  expect(Math.abs(result-expected)).toBeLessThan(2e-6)
 }
 expect(motorTorque(2,0,-2,{...legDefaults,kp:100,torqueLimit:5})).toBe(-5)
})
test('positive definite two-link inertia follows closed-form determinant at q2 zero',()=>{
 const p=legDefaults
 const s=legSnapshot([0,0,0,0],p)
 const expected=p.mass1*p.mass2*p.length1**2*p.length2**2
 expect(s.determinant).toBeCloseTo(expected,11)
 expect(s.inertia[0]).toBeGreaterThan(s.inertia[2])
 expect(s.knee[0]).toBe(0)
 expect(s.foot[1]).toBeCloseTo(p.hipHeight-p.length1-p.length2,11)
})
test('still leg with no gravity, torques or floor force remains exactly still',()=>{
 const body:PlanarLegState={joints:Float32Array.of(0,0,0,0)}
 const next=stepLegCpu(body,zeroMotor)
 expect(next.joints).toEqual(body.joints)
})
test('positive error produces positive motor acceleration with cross-coupled knee',()=>{
 const p={...zeroMotor,kp:12,kd:1,targetHip:.4,targetKnee:0}
 const next=stepLegCpu({joints:Float32Array.of(0,0,0,0)},p)
 expect(next.joints[2]).toBeGreaterThan(0)
 expect(next.joints[3]).toBeLessThan(0)
})
test('true foot penetration produces compressive normal force and friction bound',()=>{
 const p={...legDefaults,hipHeight:1.1,targetHip:0,targetKnee:0}
 const s=legSnapshot([.2,-.1,.7,-.3],p)
 expect(s.foot[1]).toBeLessThan(0)
 expect(s.normalForce).toBeGreaterThan(0)
 expect(Math.abs(s.tangentialForce)).toBeLessThanOrEqual(p.friction*s.normalForce)
})
test('passive two-link kinetic energy stays bounded over finite short integration',()=>{
 const init:PlanarLegState={joints:Float32Array.of(.3,-.4,.35,.5)}
 const initial=legSnapshot([.3,-.4,.35,.5],zeroMotor).kinetic
 let current=init
 for(let i=0;i<300;i++)current=stepLegCpu(current,{...zeroMotor,dt:.001})
 const q=Array.from(current.joints) as [number,number,number,number]
 const after=legSnapshot(q,zeroMotor).kinetic
 expect(Math.abs(after-initial)).toBeLessThan(.008)
})
test('128-leg initial ensemble evolves deterministically with finite contact diagnostics',()=>{
 let a=makeLegPopulation(128,42),b=makeLegPopulation(128,42)
 expect(a.joints).toEqual(b.joints)
 for(let i=0;i<20;i++){
  a=stepLegCpu(a)
  b=stepLegCpu(b)
 }
 expect(a.joints).toEqual(b.joints)
 const d=legPopulationDiagnostics(a)
 expect(d.count).toBe(128)
 expect(d.minimumMassDeterminant).toBeGreaterThan(0)
 expect(Number.isFinite(d.totalEnergy)).toBe(true)
 expect(Number.isFinite(d.maximumAngularSpeed)).toBe(true)
})
test('nonfinite states, nonphysical masses and oversized CPU budgets rejected',()=>{
 expect(()=>makeLegPopulation(100000)).toThrow()
 expect(()=>stepLegCpu({joints:Float32Array.of(0,0,NaN,0)})).toThrow()
 expect(()=>stepLegCpu(makeLegPopulation(4),
  {...legDefaults,mass1:0})).toThrow()
})

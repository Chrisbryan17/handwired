import {test,expect} from 'bun:test'
import {buildContactImpulseCircuit,rigidBodyDiagnostics,stepSphereContactsCpu,
  sphereContactDefaults,validateSphereBodies} from './frontiers-contacts.ts'
const make=(data:readonly number[]):{positions:Float32Array,velocities:Float32Array}=>{
 const p=new Float32Array(data.length/2),v=new Float32Array(data.length/2)
 for(let i=0;i<data.length;i+=8){
  p.set(data.slice(i,i+4),i/2)
  v.set(data.slice(i+4,i+8),i/2)
 }
 return {positions:p,velocities:v}
}
const p={...sphereContactDefaults,gravity:[0,0,0] as const,dt:0,
 restitution:1,friction:0,correctionFraction:0,iterations:1}
test('equal-mass head-on elastic contact swaps velocity, conserving momentum and energy',()=>{
 const state=make([0,0,0,1,1,0,0,1,1.9,0,0,1,-1,0,0,1])
 const before=rigidBodyDiagnostics(state)
 const result=stepSphereContactsCpu(state,p)
 expect(result.contactsProcessed).toBe(1)
 expect(result.state.velocities[0]).toBeCloseTo(-1,6)
 expect(result.state.velocities[4]).toBeCloseTo(1,6)
 const after=rigidBodyDiagnostics(result.state)
 expect(after.momentum).toEqual(before.momentum)
 expect(after.kinetic).toBeCloseTo(before.kinetic,7)
})
test('anchored sphere never moves and absorbs the moving sphere normal impulse',()=>{
 const state=make([0,0,0,1,1,0,0,1,1.8,0,0,1,0,0,0,0])
 const result=stepSphereContactsCpu(state,{...p,restitution:0,
  correctionFraction:1})
 expect(result.state.positions[4]).toBe(1.8)
 expect(result.state.velocities[4]).toBe(0)
 expect(result.state.velocities[0]).toBeCloseTo(0,6)
 expect(result.state.positions[0]).toBeLessThan(0)
})
test('Coulomb friction damps relative tangential velocity without normal momentum loss',()=>{
 const state=make([0,0,0,1,1,1,0,1,1.8,0,0,1,0,0,0,1])
 const before=rigidBodyDiagnostics(state)
 const result=stepSphereContactsCpu(state,{...p,restitution:0,friction:1})
 const after=rigidBodyDiagnostics(result.state)
 expect(after.momentum[0]).toBeCloseTo(before.momentum[0],6)
 expect(after.momentum[1]).toBeCloseTo(before.momentum[1],6)
 expect(result.state.velocities[1]).toBeLessThan(1)
 expect(result.state.velocities[5]).toBeGreaterThan(0)
 expect(after.kinetic).toBeLessThan(before.kinetic)
})
test('separating overlapped bodies correct position but do not cause new impact',()=>{
 const state=make([0,0,0,1,-1,0,0,1,1.5,0,0,1,1,0,0,1])
 const result=stepSphereContactsCpu(state,{...p,correctionFraction:1,
  penetrationSlop:0,iterations:2})
 expect(result.state.velocities[0]).toBe(-1)
 expect(result.state.velocities[4]).toBe(1)
 expect(result.state.positions[4]!-result.state.positions[0]!).toBeCloseTo(2,5)
})
test('gravity uses semi-implicit integration and zero-inverse-mass remains fixed',()=>{
 const state=make([0,2,0,.5,0,0,0,1,8,4,0,.5,0,0,0,0])
 const result=stepSphereContactsCpu(state,{...p,gravity:[0,-10,0],
  dt:.05})
 expect(result.state.velocities[1]).toBeCloseTo(-.5,6)
 expect(result.state.positions[1]).toBeCloseTo(1.975,6)
 expect(result.state.positions[5]).toBe(4)
})
test('Handwired contact kick circuit matches independently rounded signed impulse',()=>{
 const {circuit,outputWires}=buildContactImpulseCircuit()
 for(const [velocity,impulse,normal,inverseMass] of [
  [.4,2.5,-1,.5],[-2,-.3,.4,2],[1.25,.75,.8,0]]){
  const ref=Math.fround(Math.fround(velocity!)+
    Math.fround(Math.fround(Math.fround(impulse!)*Math.fround(normal!))*
    Math.fround(inverseMass!)))
  const actual=circuit.evaluateReference(Float32Array.from([
    velocity!,impulse!,normal!,inverseMass!]))[outputWires[0]!]!
  expect(Math.abs(actual-ref)).toBeLessThan(4e-7)
 }
})
test('rejects nonfinite states and invalid physical parameters',()=>{
 expect(()=>validateSphereBodies(make([0,0,0,-1,0,0,0,1]))).toThrow()
 expect(()=>stepSphereContactsCpu(make([0,0,0,1,0,0,0,1]),
  {...p,restitution:2})).toThrow()
})

import {test,expect} from 'bun:test'
import {stepBondedCpu,structuralDiagnostics,applyStructuralImpulse,
  makeBondedBridge,type BondedBody} from './fracture-model.ts'
function two(distance=2,critical=2):BondedBody{
 return {
  positions:Float32Array.of(0,0,0,0,distance,0,0,1),
  velocities:new Float32Array(8),
  endpoints:Uint32Array.of(0,1),
  bonds:Float32Array.of(1,10,0,.1,critical,1,0,1)
 }
}
test('Hooke axial force matches independent closed-form one-step displacement',()=>{
 const state=two()
 const result=stepBondedCpu(state,{dt:.05,gravity:[0,0,0]})
 expect(result.body.velocities[4]).toBeCloseTo(-.5,6)
 expect(result.body.positions[4]).toBeCloseTo(1.975,6)
 expect(result.diagnostics.intact).toBe(1)
})
test('critical tensile strain irreversibly fractures spring before force',()=>{
 const state=two(2,.4)
 const result=stepBondedCpu(state,{dt:.01,gravity:[0,0,0]})
 expect(result.body.bonds[7]).toBe(0)
 expect(result.diagnostics.events).toBe(1)
 expect(result.body.velocities[4]).toBe(0)
 const after=stepBondedCpu(result.body,{dt:.01,gravity:[0,0,0]})
 expect(after.diagnostics.events).toBe(0)
 expect(after.diagnostics.broken).toBe(1)
})
test('below-critical cyclic overstretch accumulates irreversible damage',()=>{
 const state=two(1.2,.6)
 state.bonds[5]=5
 let body=state
 for(let i=0;i<4;i++)body=stepBondedCpu(body,{dt:.04,gravity:[0,0,0]}).body
 expect(body.bonds[6]).toBeGreaterThan(0)
 expect(body.bonds[7]).toBe(1)
})
test('equal mass elastic spring preserves total momentum without gravity',()=>{
 const body:BondedBody={
  ...two(1.3),positions:Float32Array.of(0,0,0,1,1.3,0,0,1)
 }
 const after=stepBondedCpu(body,{dt:.02,gravity:[0,0,0]})
 const d=structuralDiagnostics(after.body)
 expect(Math.abs(d.momentum[0])).toBeLessThan(1e-6)
 expect(Math.abs(d.momentum[1])).toBeLessThan(1e-6)
 expect(d.kinetic).toBeGreaterThan(0)
})
test('anchored supports resist impulse; finite bridge nodes and links evolve',()=>{
 const bridge=makeBondedBridge(8,3)
 const {nodes,bonds}=structuralDiagnostics(bridge)
 expect(nodes).toBe(48)
 expect(bonds).toBeGreaterThan(nodes)
 const fixedIndex=0
 const impacted=applyStructuralImpulse(bridge,fixedIndex,[0,100,0])
 expect(impacted.velocities[1]).toBe(0)
 const loose=applyStructuralImpulse(bridge,11,[0,-5,2])
 const next=stepBondedCpu(loose,{dt:.005,gravity:[0,-9.81,0]})
 expect(next.body.positions).not.toEqual(bridge.positions)
 expect(next.diagnostics.intact).toBeGreaterThan(0)
})
test('invalid radii, topology, masses, damage law and timestep rejected',()=>{
 const state=two()
 state.endpoints[1]=4
 expect(()=>stepBondedCpu(state)).toThrow()
 const other=two()
 other.positions[7]=-1
 expect(()=>structuralDiagnostics(other)).toThrow()
 expect(()=>stepBondedCpu(two(),{dt:0,gravity:[0,0,0]})).toThrow()
})

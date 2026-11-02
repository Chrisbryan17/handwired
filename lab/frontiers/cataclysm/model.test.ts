import {test,expect} from 'bun:test'
import {poly6,spikyRadialDerivative,viscosityLaplacian,buildSphPressureCircuit,
 densityPressureCpu,makeFluidBlock,stepSphCpu,sphDiagnostics,sphDefaults,
 type SphState} from './model.ts'
test('3D SPH kernel normalization and compact support match analytical values',()=>{
 expect(poly6(0,1)).toBeCloseTo(315/(64*Math.PI),12)
 expect(poly6(1,1)).toBe(0)
 expect(poly6(4,1)).toBe(0)
 expect(spikyRadialDerivative(.5,1)).toBeLessThan(0)
 expect(spikyRadialDerivative(1,1)).toBe(0)
 expect(viscosityLaplacian(.5,1)).toBeGreaterThan(0)
 expect(viscosityLaplacian(1,1)).toBe(0)
})
test('Handwired pressure calculation matches signed independent FP32 reference',()=>{
 const {circuit,outputWires}=buildSphPressureCircuit()
 for(const density of [0.5,2.0,3.5,5.75,16]){
  const inputs=Float32Array.of(density,8,3.5)
  const actual=circuit.evaluateReference(inputs)[outputWires[0]!]!
  const expected=Math.fround(8*Math.fround(density-3.5))
  expect(Math.abs(actual-expected)).toBeLessThan(2e-6)
 }
})
test('isolated particle remains stationary under no gravity and viscosity',()=>{
 const state:SphState={positions:Float32Array.of(0,0,0,1),
 velocities:Float32Array.of(0,0,0,0)}
 const after=stepSphCpu(state,{...sphDefaults,gravity:[0,0,0]})
 expect(after.positions).toEqual(state.positions)
 expect(after.velocities).toEqual(state.velocities)
})
test('close pressure-coupled particles genuinely repel each other',()=>{
 const state:SphState={
  positions:Float32Array.of(-.2,0,0,1,.2,0,0,1),
  velocities:new Float32Array(8)
 }
 const params={...sphDefaults,h:1,restDensity:1,stiffness:10,gravity:[0,0,0] as const}
 const before=state.positions[4]!-state.positions[0]!
 const after=stepSphCpu(state,params)
 expect(after.positions[4]!-after.positions[0]!).toBeGreaterThan(before)
 expect(after.velocities[0]).toBeLessThan(0)
 expect(after.velocities[4]).toBeGreaterThan(0)
 expect(densityPressureCpu(state,params).pressure[0]).toBeGreaterThan(0)
})
test('SPH state stays finite and conservative in a reflecting tank',()=>{
 let state=makeFluidBlock(5,4,4)
 const before=sphDiagnostics(state)
 for(let step=0;step<35;step++)state=stepSphCpu(state)
 const after=sphDiagnostics(state)
 expect(after.mass).toBe(before.mass)
 expect(after.nonfinite).toBe(0)
 expect(after.maxSpeed).toBeGreaterThan(0)
 for(let i=0;i<state.positions.length;i+=4){
  for(let k=0;k<3;k++){
   expect(Number.isFinite(state.positions[i+k]!)).toBe(true)
   expect(state.positions[i+k]!).toBeGreaterThanOrEqual(sphDefaults.min[k]!)
   expect(state.positions[i+k]!).toBeLessThanOrEqual(sphDefaults.max[k]!)
  }
 }
})
test('invalid density and negative pressure stiffness are rejected',()=>{
 const data=makeFluidBlock(2,2,2)
 expect(()=>stepSphCpu(data,{...sphDefaults,stiffness:-1})).toThrow()
 expect(()=>stepSphCpu({...data,positions:Float32Array.of(NaN,0,0,1)})).toThrow()
})

import {test,expect} from 'bun:test'
import {buildUniverseCircuit,checkUniverseState,makeCollision,stepUniverseCpu,
  universeDiagnostics,defaultParams,FIXED_DT,STRIDE} from './model.ts'
test('repeatable seed and explicit positive masses',()=>{
  const a=makeCollision(128,41),b=makeCollision(128,41)
  expect(a).toEqual(b)
  expect(makeCollision(128,42)).not.toEqual(a)
  expect(checkUniverseState(a,defaultParams)).toBe(128)
  expect(()=>makeCollision(0)).toThrow()
  expect(()=>checkUniverseState(a,{gravity:-1,softening:.03})).toThrow()
})
test('Handwired compiled kick-drift is consistent with independent arithmetic',()=>{
  const {circuit,outputWires}=buildUniverseCircuit()
  const f=Math.fround
  for(const [i,p,v,a] of [[0,.2,.4,.8],[1,-.4,.3,-.2],[2,.15,-.7,.05]]){
    const src=Float32Array.from([.2,-.4,.15,.4,.3,-.7,.8,-.2,.05])
    const wires=circuit.evaluateReference(src)
    const vNew=f(f(v)+f(f(a)*f(FIXED_DT)))
    const pNew=f(f(p)+f(vNew*f(FIXED_DT)))
    expect(Math.abs(wires[outputWires[i]!]!-pNew)).toBeLessThan(2e-7)
    expect(Math.abs(wires[outputWires[3+i]!]!-vNew)).toBeLessThan(2e-7)
  }
})
test('Newton direct-force oracle preserves total mass and bounded momentum',()=>{
  const before=makeCollision(32,7)
  let current=before
  const start=universeDiagnostics(current)
  for(let i=0;i<40;i++)current=stepUniverseCpu(current)
  const after=universeDiagnostics(current)
  expect(after.mass).toBeCloseTo(start.mass,5)
  for(let axis=0;axis<3;axis++)
    expect(Math.abs(after.momentum[axis]!-start.momentum[axis]!)).toBeLessThan(.0001)
  expect(Math.abs((after.total-start.total)/start.total)).toBeLessThan(.12)
  expect(current.length).toBe(before.length)
  for(let i=0;i<current.length;i++)expect(Number.isFinite(current[i])).toBe(true)
})
test('zero gravity gives exact uniform velocity and orbital drift',()=>{
  const state=makeCollision(12,3),zero={gravity:0,softening:.02}
  const next=stepUniverseCpu(state,zero)
  for(let i=0;i<12;i++)for(let axis=0;axis<3;axis++){
    const at=i*STRIDE
    expect(next[at+4+axis]).toBe(state[at+4+axis])
    expect(Math.abs(next[at+axis]!-Math.fround(state[at+axis]!+
      Math.fround(state[at+4+axis]!*Math.fround(FIXED_DT))))).toBeLessThan(1e-7)
  }
})

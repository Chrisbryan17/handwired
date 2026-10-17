import {describe,expect,test} from 'bun:test'
import {createWebGpuEvaluator} from '../../src/runtime/webgpu.ts'
import {buildClothCircuit,handwrittenClothStep,CLOTH_SOURCE_SIZE} from './model.ts'
import {makeCloth,cloneCloth,clothSources,projectCloth,cpuClothFrame,tearNear,nearestNode} from './physics.ts'
const {circuit,outputWires}=buildClothCircuit()
const width=circuit.sourceSize+circuit.operations.length
const interaction={dragged:null,x:.5,y:.5,wind:1.2}
describe('Handwired Neural Cloth solver',()=>{
  test('independent FP32 Verlet arithmetic matches Handwired circuit',()=>{
    let s=737
    const random=()=>{s=(Math.imul(s,1664525)+1013904223)>>>0;return s/4294967296}
    for(let i=0;i<512;i++){
      const input=Float32Array.from({length:CLOTH_SOURCE_SIZE},()=>random()*2-1)
      const independent=handwrittenClothStep(input),values=circuit.evaluateReference(input)
      expect(Math.abs(independent[0]!-values[outputWires[0]]!)).toBeLessThanOrEqual(3e-7)
      expect(Math.abs(independent[1]!-values[outputWires[1]]!)).toBeLessThanOrEqual(3e-7)
    }
  })
  test('spring graph has valid indices and deterministic state',()=>{
    const a=makeCloth(13,9),b=makeCloth(13,9)
    expect(Array.from(a.state)).toEqual(Array.from(b.state))
    expect(a.springs.length).toBeGreaterThan(100)
    expect(a.pinned.some(value=>value===1)).toBe(true)
    for(const e of a.springs){
      expect(e.a).toBeGreaterThanOrEqual(0)
      expect(e.b).toBeLessThan(a.columns*a.rows)
      expect(e.rest).toBeGreaterThan(0)
    }
    expect(()=>makeCloth(2,10)).toThrow()
  })
  test('CPU batched executor, per-node oracle, and projection agree',async()=>{
    const cloth=makeCloth(11,8),inputs=clothSources(cloth,interaction)
    const backend=await createWebGpuEvaluator(circuit,{mode:'cpu',capacity:88,strategy:'specialized'})
    try{
      const values=await backend.evaluate(inputs)
      for(let i=0;i<88;i++){
        const ref=circuit.evaluateReference(inputs.subarray(i*CLOTH_SOURCE_SIZE,(i+1)*CLOTH_SOURCE_SIZE))
        for(let j=0;j<width;j++)expect(values[i*width+j]).toBe(ref[j])
      }
      const projected=projectCloth(cloth,values,width,outputWires,interaction)
      const reference=cpuClothFrame(cloth,interaction)
      expect(Array.from(projected.state)).toEqual(Array.from(reference.next.state))
    }finally{backend.destroy()}
  })
  test('host constraints pin fabric while integrating 45 stable physical steps',()=>{
    let cloth=makeCloth(15,11)
    for(let frame=0;frame<45;frame++){
      cloth=cpuClothFrame(cloth,interaction).next
      for(let i=0;i<cloth.pinned.length;i++){
        const at=i*4
        expect(Number.isFinite(cloth.state[at]!)).toBe(true)
        expect(Number.isFinite(cloth.state[at+1]!)).toBe(true)
        if(cloth.pinned[i]===1) {
          expect(cloth.state[at]).toBe(cloth.origin[i*2])
          expect(cloth.state[at+1]).toBe(cloth.origin[i*2+1])
        }
      }
    }
    expect(cloth.step).toBe(45)
  })
  test('drag controls and cutting permanently change an independent cloth copy',()=>{
    const cloth=makeCloth(11,9),other=cloneCloth(cloth)
    const e=cloth.springs[0]!
    const x=(cloth.state[e.a*4]!+cloth.state[e.b*4]!)/2
    const y=(cloth.state[e.a*4+1]!+cloth.state[e.b*4+1]!)/2
    expect(nearestNode(cloth,x,y)).not.toBeNull()
    const broken=tearNear(cloth,x,y,.015)
    expect(broken).toBeGreaterThan(0)
    expect(cloth.springs.filter(s=>!s.alive).length).toBeGreaterThan(0)
    expect(other.springs.every(s=>s.alive)).toBe(true)
    const dragged=3
    const step=cpuClothFrame(cloth,{dragged,x:.8,y:.2,wind:0}).next
    expect(step.state[dragged*4]).toBeCloseTo(.8,6)
    expect(step.state[dragged*4+1]).toBeCloseTo(.2,6)
  })
})

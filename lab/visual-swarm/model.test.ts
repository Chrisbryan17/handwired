import {describe,expect,test} from 'bun:test'
import {createWebGpuEvaluator} from '../../src/runtime/webgpu.ts'
import {buildSwarmCircuit, handwrittenStep, SOURCE_SIZE} from './model.ts'
import {buildSwarmInputs, initialPopulation, projectState, referenceFrame} from './simulation.ts'

const {circuit, outputWires} = buildSwarmCircuit()
const width = circuit.sourceSize + circuit.operations.length

describe('Neural Swarm deterministic physics and circuit', () => {
  test('same seed gives identical flock; other seed changes state', () => {
    const a = initialPopulation(384, 71), b = initialPopulation(384, 71)
    const c = initialPopulation(384, 72)
    expect(Array.from(a.state)).toEqual(Array.from(b.state))
    expect(Array.from(a.species)).toEqual(Array.from(b.species))
    expect(Array.from(a.state)).not.toEqual(Array.from(c.state))
  })
  test('reject invalid population, source length, mismatched state', () => {
    expect(() => initialPopulation(0)).toThrow()
    expect(() => initialPopulation(8193)).toThrow()
    expect(() => handwrittenStep(new Float32Array(4))).toThrow()
    expect(() => buildSwarmInputs(new Float32Array(7),new Uint8Array(3),
      {x:0.5,y:0.5,mode:'off'})).toThrow()
  })
  test('independent handwritten FP32 step agrees with Handwired reference', () => {
    let seed = 7361
    const random = () => {
      seed = (Math.imul(seed,1664525) + 1013904223) >>> 0
      return seed / 4294967296
    }
    for (let i = 0; i < 512; i++) {
      const source = Float32Array.from({length: SOURCE_SIZE}, () => random() * 2 - 1)
      const oracle = handwrittenStep(source)
      const reference = circuit.evaluateReference(source)
      for (let j = 0; j < 4; j++) {
        expect(Math.abs(reference[outputWires[j]!]! - oracle[j]!)).toBeLessThanOrEqual(3e-7)
      }
    }
  })
  test('spatial neighborhood and pointer steer, no nonfinite signals', () => {
    const population = initialPopulation(192)
    const off = buildSwarmInputs(population.state, population.species,
      {x:0.5,y:0.5,mode:'off'})
    const attract = buildSwarmInputs(population.state, population.species,
      {x:0.5,y:0.5,mode:'attract'})
    const repel = buildSwarmInputs(population.state, population.species,
      {x:0.5,y:0.5,mode:'repel'})
    expect(off.length).toBe(192*SOURCE_SIZE)
    for (const value of off) expect(Number.isFinite(value)).toBe(true)
    expect(off.some((value, i) => i%SOURCE_SIZE>=4 && i%SOURCE_SIZE<=9 && value!==0)).toBe(true)
    expect(attract.some((value, i) => i%SOURCE_SIZE>=10 && i%SOURCE_SIZE<=11 && value!==0)).toBe(true)
    expect(Array.from(attract)).not.toEqual(Array.from(repel))
  })
  test('CPU batch evaluator equals individual CPU reference on real swarm inputs', async () => {
    const population = initialPopulation(128)
    const input = buildSwarmInputs(population.state,population.species,
      {x:0.55,y:0.45,mode:'attract'})
    const backend = await createWebGpuEvaluator(circuit,{mode:'cpu',capacity:128,strategy:'specialized'})
    try {
      const results = await backend.evaluate(input)
      for(let i=0;i<128;i++){
        const expected=circuit.evaluateReference(input.subarray(i*SOURCE_SIZE,(i+1)*SOURCE_SIZE))
        for(let j=0;j<width;j++) expect(results[i*width+j]).toBe(expected[j])
      }
      const next = projectState(results,128,width,outputWires)
      const frame = referenceFrame(population.state,population.species,
        {x:0.55,y:0.45,mode:'attract'})
      expect(Array.from(next)).toEqual(Array.from(frame.state))
    } finally {backend.destroy()}
  })
  test('100-frame deterministic CPU simulation stays finite and in bounds', () => {
    const population=initialPopulation(96,444)
    let state=population.state
    for(let k=0;k<100;k++){
      state=referenceFrame(state,population.species,{x:0.6,y:0.35,mode:'attract'}).state
      for(let i=0;i<96;i++){
        const p=i*4
        expect(state[p]).toBeGreaterThanOrEqual(0.008)
        expect(state[p]).toBeLessThanOrEqual(0.992)
        expect(state[p+1]).toBeGreaterThanOrEqual(0.008)
        expect(state[p+1]).toBeLessThanOrEqual(0.992)
        expect(Number.isFinite(state[p+2]!)).toBe(true)
        expect(Number.isFinite(state[p+3]!)).toBe(true)
      }
    }
  })
})

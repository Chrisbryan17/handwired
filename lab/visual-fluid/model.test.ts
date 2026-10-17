import {describe,expect,test} from 'bun:test'
import {createWebGpuEvaluator} from '../../src/runtime/webgpu.ts'
import {buildFluidCircuit,handwrittenFluidStep,FLUID_SOURCE_SIZE,FLUID_CHANNELS} from './model.ts'
import {createFluid,cloneFluid,fluidAdvectionInputs,projectFluid,cpuFluidFrame,injectFluid} from './physics.ts'

const {circuit,outputWires}=buildFluidCircuit()
const width=circuit.sourceSize+circuit.operations.length
describe('Neural Fluid five-channel bilinear advection',()=>{
  test('independent FP32 bilinear reference matches Handwired circuit for seeded taps',()=>{
    let seed=1973
    const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296}
    for(let i=0;i<256;i++){
      const input=Float32Array.from({length:FLUID_SOURCE_SIZE},()=>random()*5-2)
      input[20]=random();input[21]=random()
      const handwritten=handwrittenFluidStep(input),reference=circuit.evaluateReference(input)
      for(let j=0;j<FLUID_CHANNELS;j++)
        expect(Math.abs(handwritten[j]!-reference[outputWires[j]!]!)).toBeLessThanOrEqual(3e-6)
    }
  })
  test('fluid seed and backtrace gather are deterministic and bounded',()=>{
    const a=createFluid(24,16),b=createFluid(24,16)
    expect(Array.from(a.fields)).toEqual(Array.from(b.fields))
    expect(()=>createFluid(0,16)).toThrow()
    const inputs=fluidAdvectionInputs(a)
    expect(inputs.length).toBe(24*16*FLUID_SOURCE_SIZE)
    for(let i=0;i<24*16;i++){
      const at=i*FLUID_SOURCE_SIZE
      expect(inputs[at+20]!).toBeGreaterThanOrEqual(0)
      expect(inputs[at+20]!).toBeLessThanOrEqual(1)
      expect(inputs[at+21]!).toBeGreaterThanOrEqual(0)
      expect(inputs[at+21]!).toBeLessThanOrEqual(1)
    }
  })
  test('CPU batch circuit agrees with scalar reference and complete fluid projection',async()=>{
    const fluid=createFluid(16,12)
    const inputs=fluidAdvectionInputs(fluid)
    const evaluator=await createWebGpuEvaluator(circuit,{mode:'cpu',capacity:192,strategy:'specialized'})
    try{
      const values=await evaluator.evaluate(inputs)
      for(let i=0;i<192;i++){
        const expected=circuit.evaluateReference(inputs.subarray(i*FLUID_SOURCE_SIZE,(i+1)*FLUID_SOURCE_SIZE))
        for(let j=0;j<width;j++)expect(values[i*width+j]).toBe(expected[j])
      }
      const projected=projectFluid(fluid,values,width,outputWires)
      const reference=cpuFluidFrame(fluid).next
      expect(Array.from(projected.fields)).toEqual(Array.from(reference.fields))
    }finally{evaluator.destroy()}
  })
  test('CPU fluid advances 14 steps with projection, no nonfinite fields',()=>{
    let fluid=createFluid(16,12)
    for(let frame=0;frame<14;frame++){
      fluid=cpuFluidFrame(fluid).next
      for(let i=0;i<fluid.fields.length;i++){
        const field=fluid.fields[i]!,channel=i%FLUID_CHANNELS
        expect(Number.isFinite(field)).toBe(true)
        if(channel>=2)expect(field).toBeGreaterThanOrEqual(0)
        expect(Math.abs(field)).toBeLessThanOrEqual(6)
      }
    }
    expect(fluid.step).toBe(14)
  })
  test('stirring modifies both dye and velocity but does not mutate a cloned state',()=>{
    const fluid=createFluid(20,16),copy=cloneFluid(fluid),original=fluid.fields.slice()
    injectFluid(fluid,.43,.45,.04,-.03,1.5)
    expect(Array.from(fluid.fields)).not.toEqual(Array.from(original))
    expect(Array.from(copy.fields)).toEqual(Array.from(original))
    expect(fluid.fields.every(Number.isFinite)).toBe(true)
  })
})

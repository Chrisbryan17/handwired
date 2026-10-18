import {test,expect} from 'bun:test'
import {genesisDefaults,makeGenesis,stepGenesisCpu} from '../genesis/model.ts'
import {stepGenesisCpuOptimized} from './reference-fast.ts'

test('fast typed-array stencil equals independent Gray-Scott reference',()=>{
  for(const [side,steps] of [[16,1],[48,40],[96,4]] as const){
    const initial=makeGenesis(side,side,12)
    let scalar=initial
    for(let i=0;i<steps;i++)scalar=stepGenesisCpu(scalar)
    const fast=stepGenesisCpuOptimized(initial,genesisDefaults,steps)
    expect(fast.length).toBe(scalar.state.length)
    let maxError=0
    for(let i=0;i<fast.length;i++)
      maxError=Math.max(maxError,Math.abs(fast[i]!-scalar.state[i]!))
    expect(maxError).toBeLessThan(1e-6)
  }
})
test('optimized baseline guards iteration count',()=>{
  const field=makeGenesis(16,16)
  expect(()=>stepGenesisCpuOptimized(field,genesisDefaults,0)).toThrow()
  expect(()=>stepGenesisCpuOptimized(field,genesisDefaults,10001)).toThrow()
})

import {test,expect} from 'bun:test'
import {buildGenesisCircuit,genesisDefaults,makeGenesis,stepGenesisCpu,
  genesisSources,genesisStatistics,validateGenesis} from './model.ts'
test('genesis seed determinism and bounded initial concentrations',()=>{
  const a=makeGenesis(48,32,18),b=makeGenesis(48,32,18),c=makeGenesis(48,32,19)
  expect(a.state).toEqual(b.state)
  expect(c.state).not.toEqual(a.state)
  validateGenesis(a)
  expect(()=>makeGenesis(2,4)).toThrow()
  expect(()=>validateGenesis(a,{...genesisDefaults,feed:-1})).toThrow()
})
test('Handwired generated integrator agrees with handwritten reaction update',()=>{
  const state=makeGenesis(24,24,12)
  const direct=stepGenesisCpu(state)
  const {circuit,outputWires}=buildGenesisCircuit()
  for(let y=0;y<24;y++)for(let x=0;x<24;x++){
    const src=genesisSources(state,genesisDefaults,x,y)
    const values=circuit.evaluateReference(src)
    const at=(y*24+x)*2
    for(const ch of [0,1]){
      const clamped=Math.max(0,Math.min(1,values[outputWires[ch]!]!))
      expect(Math.abs(clamped-direct.state[at+ch]!)).toBeLessThan(1e-6)
    }
  }
})
test('chemical diffusion evolves actual spatial state with finite solutions',()=>{
  let field=makeGenesis(48,32,5)
  const first=genesisStatistics(field)
  for(let i=0;i<80;i++)field=stepGenesisCpu(field)
  const later=genesisStatistics(field)
  expect(later.iteration).toBe(80)
  expect(later.meanV).not.toBe(first.meanV)
  expect(later.meanU).not.toBe(first.meanU)
  expect(later.spots).toBeGreaterThanOrEqual(0)
  validateGenesis(field)
})
test('no feed or kill on uniform pure-U field is a fixed point',()=>{
  const state=makeGenesis(16,16)
  state.state.fill(0)
  for(let i=0;i<state.state.length;i+=2)state.state[i]=1
  const stable=stepGenesisCpu(state,{feed:0,kill:0,diffuseU:.16,diffuseV:.08})
  expect(stable.state).toEqual(state.state)
})

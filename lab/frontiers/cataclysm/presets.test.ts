import {test,expect} from 'bun:test'
import {makeCataclysmPreset,cataclysmSettings,inspectPreset,
 type CataclysmPreset} from './presets.ts'
import {stepSphCpu} from './model.ts'
test('all physical 3D fluid initial conditions are deterministic and within numerical grid',()=>{
 const settings=cataclysmSettings(9.81,12,.12)
 for(const preset of ['dam-break','counter-current','droplet'] as CataclysmPreset[]){
  const a=makeCataclysmPreset(preset),b=makeCataclysmPreset(preset)
  expect(a.positions).toEqual(b.positions)
  expect(a.velocities).toEqual(b.velocities)
  const info=inspectPreset(preset,settings)
  expect(info.n).toBeGreaterThan(300)
  expect(info.n).toBeLessThanOrEqual(1024)
  expect(info.totalMass).toBe(info.n)
  const updated=stepSphCpu(a,settings)
  expect(updated.positions).not.toEqual(a.positions)
  for(const x of updated.positions)expect(Number.isFinite(x)).toBe(true)
 }
})
test('strong counter-current contains genuine opposite initial motion',()=>{
 const state=makeCataclysmPreset('counter-current')
 const n=state.positions.length/8
 expect(state.velocities[0]).toBeGreaterThan(0)
 expect(state.velocities[n*4]).toBeLessThan(0)
})
test('physical control bounds reject invalid and nonfinite input',()=>{
 expect(()=>cataclysmSettings(-1,12,.12)).toThrow()
 expect(()=>cataclysmSettings(9,0,.12)).toThrow()
 expect(()=>cataclysmSettings(9,12,Infinity)).toThrow()
 expect(()=>makeCataclysmPreset('fake' as CataclysmPreset)).toThrow()
})

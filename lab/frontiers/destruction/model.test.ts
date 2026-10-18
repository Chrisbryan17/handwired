import {test,expect} from 'bun:test'
import {validateSphereBodies,stepSphereContactsCpu} from '../../../src/runtime/frontiers-contacts.ts'
import {cellOfSpatialPoint} from '../../../src/runtime/frontiers-spatial.ts'
import {impactDefaults,impactSpatialGrid,makeImpactScenario,
  type ImpactPreset} from './model.ts'
test('all preview scenarios are reproducible valid GPU contact states',()=>{
 for(const preset of ['impact','cascade','swarm'] as ImpactPreset[]){
  const one=makeImpactScenario(preset),two=makeImpactScenario(preset)
  expect(one.positions).toEqual(two.positions)
  expect(one.velocities).toEqual(two.velocities)
  const count=validateSphereBodies(one)
  expect(count).toBeGreaterThan(90)
  for(let i=0;i<count;i++){
   const at=i*4
   expect(cellOfSpatialPoint(impactSpatialGrid,one.positions[at]!,
    one.positions[at+1]!,one.positions[at+2]!)).toBeGreaterThanOrEqual(0)
   expect(one.positions[at+3]!*2).toBeLessThanOrEqual(impactSpatialGrid.radius)
  }
  const step=stepSphereContactsCpu(one,impactDefaults)
  expect(step.state.positions.length).toBe(one.positions.length)
  for(const v of step.state.positions)expect(Number.isFinite(v)).toBe(true)
  for(const v of step.state.velocities)expect(Number.isFinite(v)).toBe(true)
 }
})
test('impact scenario includes dynamically moving impactor and stationary colliders',()=>{
 const state=makeImpactScenario('impact')
 let anchored=0,impactVelocity=0
 for(let i=0;i<state.positions.length/4;i++){
  if(state.velocities[i*4+3]===0)anchored++
  impactVelocity=Math.max(impactVelocity,Math.abs(state.velocities[i*4]!))
 }
 expect(anchored).toBe(81)
 expect(impactVelocity).toBeGreaterThan(10)
})

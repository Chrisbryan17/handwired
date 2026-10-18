import {spatialGridWgsl} from '../src/runtime/frontiers-spatial-gpu.ts'
import {sphDensityWgsl,sphIntegrateWgsl} from '../lab/frontiers/cataclysm/solver.ts'
import {makeFluidBlock,densityPressureCpu,stepSphCpu,sphDefaults,
 type SphState,type SphSettings} from '../lab/frontiers/cataclysm/model.ts'
import {mkdirSync,writeFileSync} from 'node:fs'
const dir=process.argv[2]??'/kaggle/working'
mkdirSync(dir,{recursive:true})
const experiments:{name:string,state:SphState,settings:SphSettings,steps:number}[]=[
 {name:'64-spatial-pressure',state:makeFluidBlock(4,4,4,.52),
  settings:{...sphDefaults,restDensity:1.7,stiffness:12},steps:1},
 {name:'80-viscosity',state:makeFluidBlock(5,4,4,.54),
  settings:{...sphDefaults,restDensity:2.0,stiffness:10,viscosity:.15},steps:4},
 {name:'120-multistep',state:makeFluidBlock(5,6,4,.51),
  settings:{...sphDefaults,restDensity:2.2,stiffness:9,viscosity:.07},steps:8},
]
const data=experiments.map(({name,state,settings,steps})=>{
 for(let i=0;i<state.positions.length/4;i++){
  const at=i*4
  state.velocities[at]=Math.fround(((i*11)%9-4)*.095)
  state.velocities[at+1]=Math.fround(((i*7)%11-5)*.065)
  state.velocities[at+2]=Math.fround(((i*5)%13-6)*.055)
 }
 const initial={positions:Array.from(state.positions),velocities:Array.from(state.velocities)}
 const reference=densityPressureCpu(state,settings)
 let current=state
 for(let i=0;i<steps;i++)current=stepSphCpu(current,settings)
 return {name,initial,settings,steps,
  field:{density:Array.from(reference.density),pressure:Array.from(reference.pressure)},
  expected:{positions:Array.from(current.positions),velocities:Array.from(current.velocities)},
  expectedMass:initial.positions.filter((_,i)=>i%4===3).reduce((a,b)=>a+b,0)}
})
for(const [name,content] of Object.entries({
 'spatial-grid.wgsl':spatialGridWgsl,
 'sph-density.wgsl':sphDensityWgsl,
 'sph-integrate.wgsl':sphIntegrateWgsl,
}))writeFileSync(dir+'/'+name,content)
writeFileSync(dir+'/sph-native-reference.json',JSON.stringify({data,
 reference:'Independent handwritten CPU O(N²) SPH oracle, FP32 states and FP64 sums',
 tolerances:{positions:.015,velocities:.015,density:.025,pressure:.15}},null,2))
console.log('Exported actual Handwired/SPH WGSL, 3 cases and independent CPU truth')

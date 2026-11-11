import {mkdirSync,writeFileSync} from 'node:fs'
import {batchedLegWgsl} from '../lab/frontiers/machines/gpu.ts'
import {makeLegPopulation,stepLegCpu,legDefaults,
 type PlanarLegParameters} from '../lab/frontiers/machines/model.ts'
const dir=process.argv[2]??'/kaggle/working'
mkdirSync(dir,{recursive:true})
const scenarios=[
 {name:'64-two-joint-contact',count:64,steps:1,seed:1009,
   p:{...legDefaults,hipHeight:1.2}},
 {name:'256-feedback-controlled-legs',count:256,steps:24,seed:445,
   p:{...legDefaults,kp:18,kd:3}},
 {name:'4096-parallel-articulated-environments',count:4096,steps:64,seed:8125,
   p:{...legDefaults,dt:.001,hipHeight:1.35}},
] satisfies {name:string,count:number,steps:number,seed:number,p:PlanarLegParameters}[]
const data=scenarios.map(({name,count,steps,seed,p})=>{
 const init=makeLegPopulation(count,seed)
 let cpu=init
 for(let i=0;i<steps;i++)cpu=stepLegCpu(cpu,p)
 return {name,count,steps,parameters:p,
  initial:Array.from(init.joints),expected:Array.from(cpu.joints)}
})
writeFileSync(dir+'/machines-articulated.wgsl',batchedLegWgsl)
writeFileSync(dir+'/machines-native-reference.json',JSON.stringify({
 oracle:'Independent CPU two-link Euler-Lagrange Lagrangian and penalty contact',
 circuit:'Handwired-generated proportional-derivative torque',
 tolerance:.018,scenarios:data},null,2))
console.log('Exported genuine Handwired WGSL and CPU reference for',data.map(x=>x.name))

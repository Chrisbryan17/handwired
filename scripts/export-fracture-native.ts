import {mkdirSync,writeFileSync} from 'node:fs'
import {bondedForceWgsl,bondedIntegrateWgsl} from '../lab/frontiers/destruction/fracture-gpu.ts'
import {makeBondedBridge,applyStructuralImpulse,stepBondedCpu,
 type BondedBody,type BondedSettings} from '../lab/frontiers/destruction/fracture-model.ts'
const out=process.argv[2]??'/kaggle/working'
mkdirSync(out,{recursive:true})
const params:BondedSettings={dt:.004,gravity:[0,-9.81,0]}
const damaged=makeBondedBridge(6,3,.65)
const fractureNode=5*3*2+1*2
damaged.positions[fractureNode*4]=Math.fround(damaged.positions[fractureNode*4]!+2.2)
const scenes:{name:string,body:BondedBody,steps:number}[]=[
 {name:'36-node-elastic-strain',body:applyStructuralImpulse(makeBondedBridge(6,3),19,[.2,-6,.5]),steps:1},
 {name:'true-critical-strain-fracture',body:damaged,steps:1},
 {name:'10-steps-damped-gravity',body:applyStructuralImpulse(makeBondedBridge(6,3),15,[2,-2,.1]),steps:10}
]
const data=scenes.map(({name,body,steps})=>{
 let expected=body,events=0
 for(let i=0;i<steps;i++){
  const result=stepBondedCpu(expected,params)
  expected=result.body
  events+=result.diagnostics.events
 }
 return {name,steps,settings:params,initial:{
  positions:Array.from(body.positions),velocities:Array.from(body.velocities),
  endpoints:Array.from(body.endpoints),bonds:Array.from(body.bonds)},
  expected:{positions:Array.from(expected.positions),velocities:Array.from(expected.velocities),
   bonds:Array.from(expected.bonds)},referenceEvents:events,
  referenceBroken:expected.bonds.filter((x,i)=>i%8===7&&x===0).length}
})
writeFileSync(out+'/fracture-bond-forces.wgsl',bondedForceWgsl)
writeFileSync(out+'/fracture-handwired-node.wgsl',bondedIntegrateWgsl)
writeFileSync(out+'/fracture-native-reference.json',JSON.stringify({
 reference:'Independent 3D Hooke Kelvin Voigt and irreversible damage CPU oracle',
 tolerance:{positions:.009,velocities:.009,damage:.009},scenes:data},null,2))
console.log('Real Handwired GPU structure shader exports and CPU truth:',data.map(x=>x.name).join(', '))

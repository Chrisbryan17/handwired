import {buildSphPressureCircuit} from '../lab/frontiers/cataclysm/model.ts'
import {emitInlineCircuitWgsl} from '../src/runtime/webgpu-inline.ts'
import {mkdirSync,writeFileSync} from 'node:fs'
const dir=process.argv[2]??'/kaggle/working'
mkdirSync(dir,{recursive:true})
const pressure=buildSphPressureCircuit()
const wgsl=`
@group(0) @binding(0) var<storage,read> source:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outPressure:array<f32>;
`+emitInlineCircuitWgsl(pressure.circuit,pressure.outputWires,'handwiredSphPressure')+`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) tid:vec3<u32>){
 if(tid.x>=arrayLength(&outPressure)){return;}
 let p=source[tid.x];
 var input:array<f32,3>;
 input[0u]=p.x;input[1u]=p.y;input[2u]=p.z;
 let signed=handwiredSphPressure(input);
 outPressure[tid.x]=max(0.0,signed[0u]);
}
`
const rows:number[][]=[],expected:number[]=[]
for(let i=0;i<256;i++){
 const density=Math.fround(.28+((i*47)%149)*.071)
 const stiffness=Math.fround(1+((i*19)%23)*.7)
 const rest=Math.fround(1.1+((i*17)%71)*.071)
 const raw=Math.fround(stiffness*Math.fround(density-rest))
 const oracle=Math.fround(Math.max(0,raw))
 const handwired=pressure.circuit.evaluateReference(
  Float32Array.of(density,stiffness,rest))[pressure.outputWires[0]!]!
 if(Math.abs(handwired-raw)>2e-5)
  throw new Error('Host circuit/reference parity mismatch at '+i)
 rows.push([density,stiffness,rest,0]);expected.push(oracle)
}
writeFileSync(dir+'/handwired-pressure.wgsl',wgsl)
writeFileSync(dir+'/handwired-pressure-reference.json',JSON.stringify({rows,expected,
 source:'Actual Handwired circuit and independent FP32 pressure equation',
 tolerance:0.0001},null,2))
console.log('Exported actual Handwired-compiled WGSL and 256 input/reference rows')

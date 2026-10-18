import {test,expect} from 'bun:test'
import {buildLightThroughputCircuit,cameraRay,dot,intersectScene,intersectSphere,
  refract,reflect,schlick,random,cosineHemisphere,renderCpuPixel,tracePathCpu,
  type Ray} from './model.ts'
test('sphere ray intersection returns exact nearest positive solution',()=>{
 const ray:Ray={origin:[0,0,0],direction:[0,0,-1]}
 const hit=intersectSphere(ray,{center:[0,0,-3],radius:1,color:[1,1,1],material:'mirror'})
 expect(hit?.t).toBeCloseTo(2,10)
 expect(hit?.normal).toEqual([0,0,1])
 const none=intersectSphere(ray,{center:[5,0,-3],radius:1,color:[1,1,1],material:'mirror'})
 expect(none).toBeNull()
})
test('refraction detects total internal reflection, Schlick physically bounded',()=>{
 const normal:[number,number,number]=[0,1,0]
 const ray:[number,number,number]=[.9,-Math.sqrt(1-.9*.9),0]
 expect(refract(ray,normal,1.5)).toBeNull()
 const mirror=reflect([0,-1,0],normal)
 expect(mirror).toEqual([0,1,0])
 expect(schlick(1,1/1.5)).toBeCloseTo(.04,10)
 expect(schlick(0,1/1.5)).toBeCloseTo(1,10)
})
test('cosine hemisphere samples remain normalized and face the normal',()=>{
 const rand=random(512)
 for(let i=0;i<200;i++){
  const direction=cosineHemisphere([0,1,0],rand(),rand())
  expect(dot(direction,direction)).toBeCloseTo(1,9)
  expect(dot(direction,[0,1,0])).toBeGreaterThanOrEqual(0)
 }
})
test('Handwired material throughput equals independent component multiplication',()=>{
 const {circuit,outputWires}=buildLightThroughputCircuit()
 for(const throughput of [[1,.5,.1],[.32,.83,.99],[0,1,.4]]){
  const material=[.42,.62,.87]
  const outputs=circuit.evaluateReference(Float32Array.from([...throughput,...material]))
  for(let k=0;k<3;k++)
   expect(Math.abs(outputs[outputWires[k]!]!-Math.fround(Math.fround(throughput[k]!)*Math.fround(material[k]!)))).toBeLessThan(2e-7)
 }
})
test('reference path integrator is deterministic, finite, nonnegative',()=>{
 const ray=cameraRay(30,24,64,48)
 const a=tracePathCpu(ray,4141),b=tracePathCpu(ray,4141)
 expect(a).toEqual(b)
 for(const value of a)expect(Number.isFinite(value)&&value>=0).toBe(true)
 const p=renderCpuPixel(20,24,64,48,32,123)
 for(const value of p)expect(Number.isFinite(value)&&value>=0).toBe(true)
})
test('emissive rectangle is intersected before ceiling',()=>{
 const ray:Ray={origin:[0,0,-2.85],direction:[0,1,0]}
 const hit=intersectScene(ray)
 expect(hit?.material).toBe('emissive')
 expect(hit?.t).toBeCloseTo(1.365,9)
 const rgb=tracePathCpu(ray,19)
 expect(rgb[0]).toBeCloseTo(18,9)
})

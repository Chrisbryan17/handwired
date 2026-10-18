import {test,expect} from 'bun:test'
import {cellOfSpatialPoint,countSpatialNeighborsCpu,seededSpatialPoints,
  validateSpatialGrid,validateSpatialPoints,type SpatialGridConfig} from './frontiers-spatial.ts'
const cfg:SpatialGridConfig={origin:[0,0,0],dimensions:[4,4,4],
 cellSize:1,radius:1}
test('cells index linearly, including domain boundaries',()=>{
 expect(validateSpatialGrid(cfg)).toBe(64)
 expect(cellOfSpatialPoint(cfg,0,0,0)).toBe(0)
 expect(cellOfSpatialPoint(cfg,1,0,0)).toBe(1)
 expect(cellOfSpatialPoint(cfg,0,1,0)).toBe(4)
 expect(cellOfSpatialPoint(cfg,0,0,1)).toBe(16)
 expect(cellOfSpatialPoint(cfg,4,0,0)).toBe(-1)
 expect(cellOfSpatialPoint(cfg,-.0001,0,0)).toBe(-1)
})
test('3D Euclidean neighbors include identical points but exclude self',()=>{
 const input=Float32Array.from([
  .5,.5,.5,1, .5,.5,.5,1, 1.5,.5,.5,1,
  .5,1.5,.5,1, .5,.5,1.5,1,
  .5,.5,1.501,1, 4,.5,.5,1])
 expect(Array.from(countSpatialNeighborsCpu(input,cfg))).toEqual([4,4,2,2,2,0,0])
})
test('nonzero displaced origin and crossing cell faces are supported',()=>{
 const box:SpatialGridConfig={origin:[-3,-7,2],dimensions:[8,8,8],cellSize:.5,radius:.5}
 const data=Float32Array.from([-2.9,-6.9,2.1,1,
  -2.6,-6.9,2.1,1,-2.1,-6.9,2.1,1])
 expect(Array.from(countSpatialNeighborsCpu(data,box))).toEqual([1,1,0])
})
test('deterministic densely occupied and zero-neighbor fields',()=>{
 const a=seededSpatialPoints(256,29)
 expect(a).toEqual(seededSpatialPoints(256,29))
 expect(a).not.toEqual(seededSpatialPoints(256,30))
 const world:SpatialGridConfig={origin:[0,0,0],dimensions:[10,10,10],
  cellSize:1,radius:.25}
 const counts=countSpatialNeighborsCpu(a,world)
 expect(counts.length).toBe(256)
 for(const count of counts)expect(Number.isInteger(count)&&count<=255).toBe(true)
})
test('rejects unsupported stencil, nonfinite input and oversized bounds',()=>{
 expect(()=>validateSpatialGrid({...cfg,radius:1.1})).toThrow()
 expect(()=>validateSpatialGrid({...cfg,dimensions:[512,512,512]})).toThrow()
 expect(()=>validateSpatialPoints(Float32Array.from([NaN,0,0,1]))).toThrow()
 expect(()=>validateSpatialPoints(Float32Array.of(1,2,3))).toThrow()
})

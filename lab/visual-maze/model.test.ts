import {describe,expect,test} from 'bun:test'
import {createWebGpuEvaluator} from '../../src/runtime/webgpu.ts'
import {buildMazeCircuit,MAZE_SOURCE_SIZE} from './model.ts'
import {beginSearch,circuitWave,cpuCircuitWave,generateMaze,mazeInputs,
  reconstructPath,referenceWave} from './engine.ts'

const {circuit,outputWires}=buildMazeCircuit()
const width=circuit.sourceSize+circuit.operations.length
describe('Neural Maze parallel BFS',()=>{
  test('seeded maze is deterministic, with valid entrance and goal',()=>{
    const a=generateMaze(47,31,400),b=generateMaze(47,31,400)
    expect(Array.from(a.walls)).toEqual(Array.from(b.walls))
    expect(a.walls[a.start]).toBe(0);expect(a.walls[a.goal]).toBe(0)
    expect(()=>generateMaze(42,31)).toThrow()
  })
  test('all 64 binary circuit inputs match a conventional boolean BFS step',()=>{
    for(let mask=0;mask<64;mask++){
      const bits=Float32Array.from({length:MAZE_SOURCE_SIZE},(_,i)=>(mask>>i)&1)
      const values=circuit.evaluateReference(bits)
      const visited=bits[0]!,pass=bits[5]!
      const active=(bits[1]!||bits[2]!||bits[3]!||bits[4]!)?1:0
      const frontier=(1-visited)*pass*active
      expect(values[outputWires[1]!]!).toBe(frontier)
      expect(values[outputWires[0]!]!).toBe(visited+frontier)
    }
  })
  test('entire seeded maze search agrees with independent conventional BFS every level',()=>{
    const maze=generateMaze(27,19,925)
    let state=beginSearch(maze)
    for(let i=0;i<2000&&!state.finished;i++){
      const reference=referenceWave(maze,state)
      const actual=cpuCircuitWave(maze,state).state
      expect(Array.from(actual.visited)).toEqual(Array.from(reference.visited))
      expect(Array.from(actual.frontier)).toEqual(Array.from(reference.frontier))
      expect(Array.from(actual.distance)).toEqual(Array.from(reference.distance))
      state=actual
    }
    expect(state.finished).toBe(true)
    expect(state.visited[maze.goal]).toBe(1)
    const path=reconstructPath(maze,state)
    expect(path[0]).toBe(maze.start)
    expect(path[path.length-1]).toBe(maze.goal)
    expect(path.length).toBe(state.distance[maze.goal]!+1)
    for(let i=1;i<path.length;i++){
      const a=path[i-1]!,b=path[i]!
      expect(maze.walls[b]).toBe(0)
      expect(Math.abs(a%maze.columns-b%maze.columns)+
        Math.abs(Math.floor(a/maze.columns)-Math.floor(b/maze.columns))).toBe(1)
    }
  })
  test('batched CPU circuit evaluator reproduces GPU-shaped cell batch',async()=>{
    const maze=generateMaze(31,21,915),state=beginSearch(maze)
    const input=mazeInputs(maze,state)
    const backend=await createWebGpuEvaluator(circuit,{mode:'cpu',
      capacity:maze.walls.length,strategy:'specialized'})
    try{
      const result=await backend.evaluate(input)
      for(let i=0;i<maze.walls.length;i++){
        const local=circuit.evaluateReference(input.subarray(i*MAZE_SOURCE_SIZE,(i+1)*MAZE_SOURCE_SIZE))
        for(let j=0;j<width;j++)expect(result[i*width+j]).toBe(local[j])
      }
      const actual=circuitWave(maze,state,result,width,outputWires)
      expect(Array.from(actual.visited)).toEqual(Array.from(referenceWave(maze,state).visited))
    }finally{backend.destroy()}
  })
})

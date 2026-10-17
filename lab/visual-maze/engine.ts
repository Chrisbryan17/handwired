import {buildMazeCircuit, MAZE_SOURCE_SIZE} from './model.ts'
export type Maze={columns:number,rows:number,walls:Uint8Array,start:number,goal:number}
export type SearchState={visited:Uint8Array,frontier:Uint8Array,distance:Int32Array,depth:number,finished:boolean}
const directions=[[0,-1],[1,0],[0,1],[-1,0]] as const
const position=(cols:number,x:number,y:number)=>y*cols+x
export function generateMaze(columns=47,rows=31,seed=215){
  if(!Number.isSafeInteger(columns)||!Number.isSafeInteger(rows)||columns<5||rows<5||
    columns%2!==1||rows%2!==1||columns*rows>65536)throw new Error('maze must have odd dimensions >=5')
  let state=seed>>>0
  const random=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return state/4294967296}
  const walls=new Uint8Array(columns*rows).fill(1)
  const start=position(columns,1,1),goal=position(columns,columns-2,rows-2)
  const stack=[start];walls[start]=0
  while(stack.length){
    const current=stack[stack.length-1]!,cx=current%columns,cy=Math.floor(current/columns)
    const options:{x:number,y:number,mx:number,my:number}[]=[]
    for(const [dx,dy] of directions){
      const x=cx+dx*2,y=cy+dy*2
      if(x<1||x>=columns-1||y<1||y>=rows-1||walls[position(columns,x,y)]===0)continue
      options.push({x,y,mx:cx+dx,my:cy+dy})
    }
    if(options.length===0){stack.pop();continue}
    const next=options[Math.floor(random()*options.length)]!
    walls[position(columns,next.mx,next.my)]=0
    walls[position(columns,next.x,next.y)]=0
    stack.push(position(columns,next.x,next.y))
  }
  return {columns,rows,walls,start,goal} satisfies Maze
}
export function beginSearch(maze:Maze):SearchState {
  const {walls,start}=maze
  if(walls[start]!==0)throw new Error('start must be walkable')
  const visited=new Uint8Array(walls.length),frontier=new Uint8Array(walls.length)
  const distance=new Int32Array(walls.length).fill(-1)
  visited[start]=1;frontier[start]=1;distance[start]=0
  return {visited,frontier,distance,depth:0,finished:start===maze.goal}
}
const adjacent=(maze:Maze,index:number):number[]=>{
  const x=index%maze.columns,y=Math.floor(index/maze.columns),out:number[]=[]
  for(const [dx,dy] of directions){
    const xx=x+dx,yy=y+dy
    if(xx>=0&&xx<maze.columns&&yy>=0&&yy<maze.rows)
      out.push(position(maze.columns,xx,yy))
  }
  return out
}
export function mazeInputs(maze:Maze,state:SearchState):Float32Array {
  const count=maze.walls.length
  if(state.visited.length!==count||state.frontier.length!==count)throw new Error('maze state size mismatch')
  const input=new Float32Array(count*MAZE_SOURCE_SIZE)
  for(let i=0;i<count;i++){
    const x=i%maze.columns,y=Math.floor(i/maze.columns),at=i*MAZE_SOURCE_SIZE
    input[at]=state.visited[i]!
    for(const [j,[dx,dy]] of directions.entries()){
      const xx=x+dx,yy=y+dy
      input[at+1+j]=(xx<0||yy<0||xx>=maze.columns||yy>=maze.rows)
        ? 0 : state.frontier[position(maze.columns,xx,yy)]!
    }
    input[at+5]=maze.walls[i]===0?1:0
  }
  return input
}
/** Independent conventional BFS-level expansion, no circuit operations. */
export function referenceWave(maze:Maze,state:SearchState):SearchState {
  if(state.finished)return state
  const visited=state.visited.slice(),frontier=new Uint8Array(maze.walls.length)
  const distance=state.distance.slice(),depth=state.depth+1
  let advanced=false
  for(let i=0;i<maze.walls.length;i++){
    if(state.frontier[i]===0)continue
    for(const other of adjacent(maze,i)){
      if(maze.walls[other]!==0||visited[other]===1)continue
      visited[other]=1;frontier[other]=1;distance[other]=depth;advanced=true
    }
  }
  return {visited,frontier,distance,depth,finished:visited[maze.goal]===1||!advanced}
}
/** Circuit batches supply visited and new-frontier wires. */
export function circuitWave(maze:Maze,state:SearchState,values:Float32Array,
  width:number,wires:readonly[number,number]):SearchState {
  const count=maze.walls.length
  if(values.length!==count*width)throw new Error('GPU batch length mismatch')
  const visited=new Uint8Array(count),frontier=new Uint8Array(count),distance=state.distance.slice()
  let advanced=false
  for(let i=0;i<count;i++){
    const v=values[i*width+wires[0]]!,f=values[i*width+wires[1]]!
    if(!Number.isFinite(v)||!Number.isFinite(f)||Math.abs(v-Math.round(v))>2e-5||
      Math.abs(f-Math.round(f))>2e-5)throw new Error('circuit produced non-bit BFS state')
    visited[i]=v>=0.5?1:0
    frontier[i]=f>=0.5?1:0
    if(frontier[i]===1){advanced=true;distance[i]=state.depth+1}
  }
  return {visited,frontier,distance,depth:state.depth+1,
    finished:visited[maze.goal]===1||!advanced}
}
export function reconstructPath(maze:Maze,state:SearchState):number[] {
  if(state.distance[maze.goal]===-1)return []
  let at=maze.goal
  const path=[at]
  while(at!==maze.start){
    const d=state.distance[at]!
    const next=adjacent(maze,at).find(i=>state.distance[i]===d-1)
    if(next===undefined)throw new Error('inconsistent predecessor distances')
    path.push(next);at=next
  }
  return path.reverse()
}
export function cpuCircuitWave(maze:Maze,state:SearchState) {
  const {circuit,outputWires}=buildMazeCircuit()
  const input=mazeInputs(maze,state),width=circuit.sourceSize+circuit.operations.length
  const values=new Float32Array(maze.walls.length*width)
  for(let i=0;i<maze.walls.length;i++){
    values.set(circuit.evaluateReference(input.subarray(i*MAZE_SOURCE_SIZE,(i+1)*MAZE_SOURCE_SIZE)),i*width)
  }
  return {input,values,state:circuitWave(maze,state,values,width,outputWires)}
}

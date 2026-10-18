/**
 * Shared three-dimensional uniform spatial hashing contract.
 *
 * Deterministic independent all-pairs CPU oracle for GPU linked-cell queries.
 * Radius must be no larger than a cell edge: searching the current cell and
 * the 26 neighboring cells then covers every possible Euclidean neighbor.
 * World positions outside the configured domain are excluded on both paths.
 */
export type Grid3 = readonly [number,number,number]
export type SpatialGridConfig = {
  origin:Grid3
  dimensions:readonly [number,number,number]
  cellSize:number
  radius:number
}
export function validateSpatialGrid(config:SpatialGridConfig):number {
  const {origin,dimensions,cellSize,radius}=config
  if(!origin.every(Number.isFinite) || !dimensions.every(x=>Number.isSafeInteger(x)&&x>=1&&x<=512) ||
    !Number.isFinite(cellSize)||cellSize<=0||cellSize>1e7||
    !Number.isFinite(radius)||radius<=0||radius>cellSize)
    throw new Error('invalid spatial-grid bounds or unsupported radius')
  const cells=dimensions[0]*dimensions[1]*dimensions[2]
  if(!Number.isSafeInteger(cells)||cells>4_194_304)
    throw new Error('spatial grid exceeds supported cell budget')
  for(let i=0;i<3;i++){
    if(!Number.isFinite(origin[i]+dimensions[i]*cellSize))
      throw new Error('spatial-grid world extent overflow')
  }
  return cells
}
export function validateSpatialPoints(points:Float32Array):number {
  if(points.length===0||points.length%4!==0||points.length/4>1_048_576)
    throw new Error('points must be packed as nonempty xyz+w float32 tuples')
  for(const value of points){
    if(!Number.isFinite(value))throw new Error('nonfinite spatial point component')
  }
  return points.length/4
}
export function cellOfSpatialPoint(config:SpatialGridConfig,
  x:number,y:number,z:number):number {
  const [nx,ny,nz]=config.dimensions
  const ix=Math.floor((x-config.origin[0])/config.cellSize)
  const iy=Math.floor((y-config.origin[1])/config.cellSize)
  const iz=Math.floor((z-config.origin[2])/config.cellSize)
  return ix>=0&&iy>=0&&iz>=0&&ix<nx&&iy<ny&&iz<nz
    ?(iz*ny+iy)*nx+ix:-1
}
export function countSpatialNeighborsCpu(points:Float32Array,
  config:SpatialGridConfig):Uint32Array {
  validateSpatialGrid(config)
  const n=validateSpatialPoints(points),result=new Uint32Array(n)
  const valid=new Uint8Array(n)
  for(let i=0;i<n;i++)valid[i]=Number(cellOfSpatialPoint(config,points[i*4]!,
    points[i*4+1]!,points[i*4+2]!)>=0)
  const radiusSq=config.radius*config.radius
  // Deliberately all-pairs, not a copy of linked-cell GPU algorithm.
  for(let i=0;i<n;i++){
    if(valid[i]===0)continue
    for(let j=0;j<n;j++){
      if(j===i||valid[j]===0)continue
      const dx=points[i*4]!-points[j*4]!
      const dy=points[i*4+1]!-points[j*4+1]!
      const dz=points[i*4+2]!-points[j*4+2]!
      if(dx*dx+dy*dy+dz*dz<=radiusSq)result[i]!++
    }
  }
  return result
}
export function seededSpatialPoints(count:number,seed=43):Float32Array {
  if(!Number.isSafeInteger(count)||count<1||count>100_000)
    throw new Error('invalid point count')
  let state=seed>>>0
  const next=()=>{
    state=(Math.imul(1664525,state)+1013904223)>>>0
    return state/4294967296
  }
  const data=new Float32Array(count*4)
  for(let i=0;i<count;i++){
    data[i*4]=Math.fround(next()*9.8+.1)
    data[i*4+1]=Math.fround(next()*9.8+.1)
    data[i*4+2]=Math.fround(next()*9.8+.1)
    data[i*4+3]=1
  }
  return data
}

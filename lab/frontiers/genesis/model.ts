import {Circuit} from '../../../src/runtime/numeric-ui-core.ts'
/**
 * Reaction–diffusion morphogenesis substrate, Gray–Scott chemistry.
 * A deterministic programmable morphogen field, NOT a learned neural
 * cellular automaton and NOT evolution or reproduction.
 */
export type GenesisParams={feed:number,kill:number,diffuseU:number,diffuseV:number}
export const genesisDefaults:GenesisParams={feed:.035,kill:.062,diffuseU:.16,diffuseV:.08}
export type GenesisField={width:number,height:number,state:Float32Array,iteration:number}
const stride=2
const f=Math.fround
const clamp01=(x:number)=>Math.min(1,Math.max(0,x))
export function buildGenesisCircuit(){
  // [u,v, lapU,lapV, uvv, feed*(1-u), (feed+kill)*v]
  const c=new Circuit(7)
  const s=Array.from({length:7},(_,i)=>c.source(i))
  const u=c.roundAffine([[s[0]!,1],[s[2]!,genesisDefaults.diffuseU],
    [s[4]!,-1],[s[5]!,1]])
  const v=c.roundAffine([[s[1]!,1],[s[3]!,genesisDefaults.diffuseV],
    [s[4]!,1],[s[6]!,-1]])
  return {circuit:c,outputWires:[u,v] as const}
}
function seededRandom(seed:number){
  let value=seed>>>0
  return()=>{value=(Math.imul(value,1664525)+1013904223)>>>0;return value/4294967296}
}
export function makeGenesis(width=96,height=96,seed=7):GenesisField{
  if(!Number.isSafeInteger(width)||!Number.isSafeInteger(height)||
    width<8||height<8||width*height>4_194_304||
    !Number.isSafeInteger(seed))throw new Error('invalid morphogen grid or seed')
  const state=new Float32Array(width*height*stride)
  const random=seededRandom(seed)
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const at=(y*width+x)*2
    let u=1,v=0
    const dx=(x-width/2)/Math.min(width,height)
    const dy=(y-height/2)/Math.min(width,height)
    if(dx*dx+dy*dy<.009 || (random()<.002)){
      u=.5+random()*.05
      v=.23+random()*.07
    }
    state[at]=u;state[at+1]=v
  }
  return {width,height,state,iteration:0}
}
export function validateGenesis(field:GenesisField,params:GenesisParams=genesisDefaults){
  if(!Number.isSafeInteger(field.width)||!Number.isSafeInteger(field.height)||
    field.width<8||field.height<8||field.width*field.height*2!==field.state.length)
    throw new Error('invalid morphogen field dimensions')
  if(!Number.isSafeInteger(field.iteration)||field.iteration<0)
    throw new Error('invalid morphogen iteration')
  if(![params.feed,params.kill,params.diffuseU,params.diffuseV].every(Number.isFinite)||
    params.feed<0||params.feed>.12||params.kill<0||params.kill>.12||
    params.diffuseU<0||params.diffuseU>.22||params.diffuseV<0||params.diffuseV>.22)
    throw new Error('invalid Gray-Scott coefficients')
  for(let i=0;i<field.state.length;i++){
    const value=field.state[i]!
    if(!Number.isFinite(value)||value<0||value>1)
      throw new Error('invalid morphogen concentration')
  }
}
export function genesisSources(field:GenesisField,params:GenesisParams=genesisDefaults,
  x:number,y:number):Float32Array{
  const {width,height,state}=field
  const wrap=(coord:number,size:number)=>(coord%size+size)%size
  const sample=(dx:number,dy:number,ch:number)=>
    state[((wrap(y+dy,height)*width)+wrap(x+dx,width))*2+ch]!
  const lap=(ch:number)=>{
    let value=-sample(0,0,ch)
    for(const [dx,dy] of [[-1,0],[1,0],[0,-1],[0,1]])
      value+=.2*sample(dx!,dy!,ch)
    for(const [dx,dy] of [[-1,-1],[-1,1],[1,-1],[1,1]])
      value+=.05*sample(dx!,dy!,ch)
    return value
  }
  const u=sample(0,0,0),v=sample(0,0,1)
  const uvv=u*v*v
  // Diffusion scaling is a shader-side source transform, not a hidden
  // Handwired operation. The generated circuit includes default constants.
  return Float32Array.of(u,v,
    lap(0)*params.diffuseU/genesisDefaults.diffuseU,
    lap(1)*params.diffuseV/genesisDefaults.diffuseV,
    uvv,params.feed*(1-u),(params.feed+params.kill)*v)
}
/** Entirely independent CPU Gray-Scott finite-difference oracle. */
export function stepGenesisCpu(field:GenesisField,
  params:GenesisParams=genesisDefaults):GenesisField{
  validateGenesis(field,params)
  const {width,height,state}=field
  const next=new Float32Array(state.length)
  const wrap=(value:number,size:number)=>(value+size)%size
  const read=(x:number,y:number,c:number)=>state[(wrap(y,height)*width+wrap(x,width))*2+c]!
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const i=(y*width+x)*2,u=state[i]!,v=state[i+1]!
    let lapU=-u,lapV=-v
    for(const [dx,dy,weight] of [[-1,0,.2],[1,0,.2],[0,-1,.2],[0,1,.2],
      [-1,-1,.05],[1,-1,.05],[-1,1,.05],[1,1,.05]]){
      lapU+=weight!*read(x+dx!,y+dy!,0)
      lapV+=weight!*read(x+dx!,y+dy!,1)
    }
    const reaction=u*v*v
    next[i]=f(clamp01(u+params.diffuseU*lapU-reaction+params.feed*(1-u)))
    next[i+1]=f(clamp01(v+params.diffuseV*lapV+reaction-(params.feed+params.kill)*v))
  }
  return {width,height,state:next,iteration:field.iteration+1}
}
export function genesisStatistics(field:GenesisField){
  validateGenesis(field)
  let u=0,v=0,spots=0
  for(let i=0;i<field.state.length;i+=2){
    u+=field.state[i]!;v+=field.state[i+1]!
    if(field.state[i+1]!>.18)spots++
  }
  return {meanU:u/(field.width*field.height),meanV:v/(field.width*field.height),
    spots,iteration:field.iteration}
}

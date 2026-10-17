import {Circuit,type Wire} from '../../src/runtime/numeric-ui-core.ts'

/**
 * Five-channel semi-Lagrangian advection through a fixed neural computation graph.
 * Host computes backtraced coordinates and gathers each channel's four taps.
 * Handwired computes all bilinear interpolation products and sums (CPU or WebGPU).
 * Pressure projection, diffusion and mouse forces remain host-side.
 */
export const FLUID_CHANNELS=5
export const FLUID_SOURCE_SIZE=22
export function buildFluidCircuit(){
  const c=new Circuit(FLUID_SOURCE_SIZE)
  const tx=c.source(20),ty=c.source(21)
  c.assumeRange(tx,0,1);c.assumeRange(ty,0,1)
  const lerp=(a:Wire,b:Wire,t:Wire)=>{
    const delta=c.roundAffine([[b,1],[a,-1]])
    const product=c.reglu([[delta,1]],0,[[t,1]],0) // t in [0,1]
    return c.roundAffine([[a,1],[product,1]])
  }
  const outputWires:Array<Wire>=[]
  for(let channel=0;channel<FLUID_CHANNELS;channel++){
    const k=channel*4
    const top=lerp(c.source(k),c.source(k+1),tx)
    const bottom=lerp(c.source(k+2),c.source(k+3),tx)
    outputWires.push(lerp(top,bottom,ty))
  }
  return {circuit:c,outputWires:outputWires as [Wire,Wire,Wire,Wire,Wire]}
}
export function handwrittenFluidStep(input:Float32Array):Float32Array {
  if(input.length!==FLUID_SOURCE_SIZE)throw new Error('expected 22 advection sources')
  const f=Math.fround,tx=input[20]!,ty=input[21]!
  const lerp=(a:number,b:number,t:number)=>f(f(a)+f(f(f(b)-f(a))*f(t)))
  return Float32Array.from({length:FLUID_CHANNELS},(_,channel)=>{
    const k=channel*4
    return lerp(lerp(input[k]!,input[k+1]!,tx),
      lerp(input[k+2]!,input[k+3]!,tx),ty)
  })
}

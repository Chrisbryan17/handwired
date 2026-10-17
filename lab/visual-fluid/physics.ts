import {buildFluidCircuit,FLUID_CHANNELS,FLUID_SOURCE_SIZE} from './model.ts'

export type Fluid={width:number,height:number,fields:Float32Array,step:number}
export const FLUID_DT=.48
const clamp=(x:number,min:number,max:number)=>Math.min(max,Math.max(min,x))
const index=(fluid:Fluid,x:number,y:number)=>((y*fluid.width)+x)*FLUID_CHANNELS
export function createFluid(width=48,height=32):Fluid {
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<8||height<8||
    width*height>8192)throw new Error('invalid fluid dimensions')
  const fluid:Fluid={width,height,fields:new Float32Array(width*height*FLUID_CHANNELS),step:0}
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const dx=(x-width*.5)/width,dy=(y-height*.5)/height
    const at=index(fluid,x,y),d=dx*dx+dy*dy
    const swirl=Math.exp(-d*10)
    fluid.fields[at]=-dy*4.5*swirl
    fluid.fields[at+1]=dx*4.5*swirl
    const a=Math.exp(-((dx+.19)**2+(dy+.08)**2)*110)
    const b=Math.exp(-((dx-.17)**2+(dy-.12)**2)*150)
    const c=Math.exp(-((dx+.02)**2+(dy-.21)**2)*170)
    fluid.fields[at+2]=a*2.5+b*.2+c*.4
    fluid.fields[at+3]=a*.15+b*1.9+c*.6
    fluid.fields[at+4]=a*.8+b*3.1+c*2.2
  }
  return fluid
}
export function cloneFluid(fluid:Fluid):Fluid {
  return {...fluid,fields:fluid.fields.slice()}
}
/** Mouse input feeds BOTH physics lanes; dye injection and momentum are host operations. */
export function injectFluid(fluid:Fluid,x:number,y:number,dx:number,dy:number,strength=1){
  const cx=x*fluid.width,cy=y*fluid.height
  for(let j=Math.max(1,Math.floor(cy-5));j<Math.min(fluid.height-1,Math.ceil(cy+5));j++)
    for(let i=Math.max(1,Math.floor(cx-5));i<Math.min(fluid.width-1,Math.ceil(cx+5));i++){
      const d=(i-cx)**2+(j-cy)**2,weight=Math.exp(-d/9)*strength
      const at=index(fluid,i,j)
      fluid.fields[at]=clamp(fluid.fields[at]!+dx*weight*8,-3,3)
      fluid.fields[at+1]=clamp(fluid.fields[at+1]!+dy*weight*8,-3,3)
      const hue=(fluid.step*.035)%6.283
      fluid.fields[at+2]=clamp(fluid.fields[at+2]!+weight*(.3+.7*(1+Math.sin(hue)) ),0,6)
      fluid.fields[at+3]=clamp(fluid.fields[at+3]!+weight*(.3+.6*(1+Math.sin(hue+2.1))),0,6)
      fluid.fields[at+4]=clamp(fluid.fields[at+4]!+weight*(.3+.6*(1+Math.sin(hue+4.2))),0,6)
    }
}
/** Semi-Lagrangian backtrace + bilinear neighborhood gather. */
export function fluidAdvectionInputs(fluid:Fluid):Float32Array {
  const {width,height,fields}=fluid
  const sources=new Float32Array(width*height*FLUID_SOURCE_SIZE)
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const cell=(y*width+x),input=cell*FLUID_SOURCE_SIZE,at=cell*FLUID_CHANNELS
    const bx=clamp(x-fields[at]!*FLUID_DT,0,width-1.000001)
    const by=clamp(y-fields[at+1]!*FLUID_DT,0,height-1.000001)
    const x0=Math.floor(bx),y0=Math.floor(by)
    const x1=Math.min(width-1,x0+1),y1=Math.min(height-1,y0+1)
    sources[input+20]=bx-x0;sources[input+21]=by-y0
    for(let channel=0;channel<FLUID_CHANNELS;channel++){
      const k=input+channel*4
      sources[k]=fields[index(fluid,x0,y0)+channel]!
      sources[k+1]=fields[index(fluid,x1,y0)+channel]!
      sources[k+2]=fields[index(fluid,x0,y1)+channel]!
      sources[k+3]=fields[index(fluid,x1,y1)+channel]!
    }
  }
  return sources
}
export function projectFluid(fluid:Fluid,values:Float32Array,width:number,outputWires:readonly[number,number,number,number,number]):Fluid {
  const count=fluid.width*fluid.height
  if(values.length!==count*width)throw new Error('fluid output batch mismatch')
  const next=cloneFluid(fluid),fields=next.fields
  for(let i=0;i<count;i++)for(let k=0;k<FLUID_CHANNELS;k++){
    const value=values[i*width+outputWires[k]!]!
    if(!Number.isFinite(value))throw new Error('nonfinite advection result')
    fields[i*FLUID_CHANNELS+k]=clamp(value,k<2?-3:0,k<2?3:6)
  }
  // Mild explicit diffusion for all five channels (host-side five-point stencil).
  const advected=fields.slice(),W=fluid.width,H=fluid.height
  const sample=(x:number,y:number,k:number)=>
    advected[index(fluid,clamp(x,0,W-1),clamp(y,0,H-1))+k]!
  for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++)for(let k=0;k<5;k++){
    const at=index(fluid,x,y)+k
    const average=.25*(sample(x-1,y,k)+sample(x+1,y,k)+sample(x,y-1,k)+sample(x,y+1,k))
    fields[at]=clamp(advected[at]!+(average-advected[at]!)*(k<2?.016:.006),k<2?-3:0,k<2?3:6)
  }
  // Pressure projection: divergence and Jacobi pressure Poisson solve.
  const divergence=new Float32Array(count),pressure=new Float32Array(count)
  let scratch=new Float32Array(count)
  const vel=(x:number,y:number,c:number)=>fields[index(fluid,clamp(x,0,W-1),clamp(y,0,H-1))+c]!
  for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){
    divergence[y*W+x]=-.5*(vel(x+1,y,0)-vel(x-1,y,0)+vel(x,y+1,1)-vel(x,y-1,1))
  }
  for(let iter=0;iter<20;iter++){
    for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){
      const i=y*W+x
      scratch[i]=.25*(pressure[i-1]!+pressure[i+1]!+pressure[i-W]!+pressure[i+W]!+divergence[i]!)
    }
    pressure.set(scratch);scratch=new Float32Array(count)
  }
  for(let y=1;y<H-1;y++)for(let x=1;x<W-1;x++){
    const i=y*W+x,at=i*FLUID_CHANNELS
    fields[at]=clamp(fields[at]!-.5*(pressure[i+1]!-pressure[i-1]!),-3,3)
    fields[at+1]=clamp(fields[at+1]!-.5*(pressure[i+W]!-pressure[i-W]!),-3,3)
    for(let k=2;k<5;k++)fields[at+k]=clamp(fields[at+k]!*.998,0,6)
  }
  // No penetration through the exterior domain.
  for(let x=0;x<W;x++)for(const y of [0,H-1]){
    const i=index(fluid,x,y);fields[i]=0;fields[i+1]=0
  }
  for(let y=0;y<H;y++)for(const x of [0,W-1]){
    const i=index(fluid,x,y);fields[i]=0;fields[i+1]=0
  }
  next.step++
  return next
}
export function cpuFluidFrame(fluid:Fluid) {
  const {circuit,outputWires}=buildFluidCircuit()
  const sources=fluidAdvectionInputs(fluid)
  const width=circuit.sourceSize+circuit.operations.length
  const values=new Float32Array(fluid.width*fluid.height*width)
  for(let i=0;i<fluid.width*fluid.height;i++){
    values.set(circuit.evaluateReference(sources.subarray(i*FLUID_SOURCE_SIZE,(i+1)*FLUID_SOURCE_SIZE)),i*width)
  }
  return {sources,values,next:projectFluid(fluid,values,width,outputWires)}
}

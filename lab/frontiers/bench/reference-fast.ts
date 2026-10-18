import type {GenesisField,GenesisParams} from '../genesis/model.ts'

/** CPU baseline: periodic stencil topology precomputed once, no per-cell
 * modulo, closure or temporary array allocation during simulation. */
export function stepGenesisCpuOptimized(field:GenesisField,
  parameters:GenesisParams,steps:number):Float32Array {
  const {width,height}=field
  if(!Number.isSafeInteger(steps)||steps<=0||steps>10000)
    throw new Error('invalid optimized morphogen iteration count')
  const cells=width*height,neighbor=new Uint32Array(cells*8)
  for(let y=0;y<height;y++)for(let x=0;x<width;x++){
    const x0=x===0?width-1:x-1,x1=x===width-1?0:x+1
    const y0=y===0?height-1:y-1,y1=y===height-1?0:y+1
    const base=(y*width+x)*8
    neighbor[base]=y*width+x0;neighbor[base+1]=y*width+x1
    neighbor[base+2]=y0*width+x;neighbor[base+3]=y1*width+x
    neighbor[base+4]=y0*width+x0;neighbor[base+5]=y0*width+x1
    neighbor[base+6]=y1*width+x0;neighbor[base+7]=y1*width+x1
  }
  let src=field.state.slice(),dst=new Float32Array(field.state.length)
  for(let step=0;step<steps;step++){
    for(let index=0;index<cells;index++){
      const at=index*2,base=index*8,u=src[at]!,v=src[at+1]!
      let lapU=-u,lapV=-v
      for(let k=0;k<8;k++){
        const j=neighbor[base+k]!*2,w=k<4?.2:.05
        lapU+=w*src[j]!
        lapV+=w*src[j+1]!
      }
      const reaction=u*v*v
      dst[at]=Math.fround(Math.max(0,Math.min(1,
        u+parameters.diffuseU*lapU-reaction+parameters.feed*(1-u))))
      dst[at+1]=Math.fround(Math.max(0,Math.min(1,
        v+parameters.diffuseV*lapV+reaction-(parameters.feed+parameters.kill)*v)))
    }
    const temp=src;src=dst;dst=temp
  }
  return src
}

import {buildClothCircuit,CLOTH_SOURCE_SIZE} from './model.ts'

export type Spring={a:number,b:number,rest:number,alive:boolean}
export type Cloth={
  columns:number,rows:number,state:Float32Array,origin:Float32Array,
  pinned:Uint8Array,springs:Spring[],step:number
}
export type ClothInteraction={dragged:number|null,x:number,y:number,wind:number}
const clamp=(x:number,min:number,max:number)=>Math.max(min,Math.min(max,x))
export function makeCloth(columns=25,rows=17):Cloth {
  if(!Number.isSafeInteger(columns)||!Number.isSafeInteger(rows)||
    columns<3||rows<3||columns*rows>4096)throw new Error('invalid cloth mesh dimensions')
  const count=columns*rows,state=new Float32Array(count*4),origin=new Float32Array(count*2)
  const pinned=new Uint8Array(count),springs:Spring[]=[]
  const dx=.78/(columns-1),dy=.59/(rows-1)
  for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
    const i=y*columns+x,px=.11+x*dx,py=.08+y*dy
    state.set([px,py,px,py],i*4);origin.set([px,py],i*2)
    if(y===0 && (x===0 || x===columns-1 || x%5===0))pinned[i]=1
  }
  const spring=(a:number,b:number)=>{
    const dx=origin[a*2]!-origin[b*2]!,dy=origin[a*2+1]!-origin[b*2+1]!
    springs.push({a,b,rest:Math.hypot(dx,dy),alive:true})
  }
  for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
    const i=y*columns+x
    if(x+1<columns)spring(i,i+1)
    if(y+1<rows)spring(i,i+columns)
    if(x+1<columns&&y+1<rows){spring(i,i+columns+1);spring(i+1,i+columns)}
  }
  return {columns,rows,state,origin,pinned,springs,step:0}
}
export function cloneCloth(cloth:Cloth):Cloth{
  return {...cloth,state:cloth.state.slice(),origin:cloth.origin.slice(),
    pinned:cloth.pinned.slice(),springs:cloth.springs.map(s=>({...s}))}
}
export function clothSources(cloth:Cloth,interaction:ClothInteraction):Float32Array{
  const count=cloth.columns*cloth.rows,inputs=new Float32Array(count*CLOTH_SOURCE_SIZE)
  for(let i=0;i<count;i++){
    const at=i*4,dest=i*CLOTH_SOURCE_SIZE
    inputs.set(cloth.state.subarray(at,at+4),dest)
    const x=cloth.state[at]!,y=cloth.state[at+1]!
    inputs[dest+6]=interaction.wind*Math.sin(cloth.step*.065+x*12+y*5)*0.65
    inputs[dest+7]=0.10
  }
  // Hooke axial + shear springs, bounded per-vertex force. Host stage is explicit.
  for(const edge of cloth.springs){
    if(!edge.alive)continue
    const a=edge.a*4,b=edge.b*4,ax=cloth.state[a]!,ay=cloth.state[a+1]!
    const dx=cloth.state[b]!-ax,dy=cloth.state[b+1]!-ay
    const distance=Math.hypot(dx,dy)
    if(distance<1e-8)continue
    const strength=clamp((distance-edge.rest)*3.2,-.6,.6)/distance
    const fx=dx*strength,fy=dy*strength
    inputs[edge.a*CLOTH_SOURCE_SIZE+4]+=fx
    inputs[edge.a*CLOTH_SOURCE_SIZE+5]+=fy
    inputs[edge.b*CLOTH_SOURCE_SIZE+4]-=fx
    inputs[edge.b*CLOTH_SOURCE_SIZE+5]-=fy
  }
  for(let i=0;i<count;i++){
    const dest=i*CLOTH_SOURCE_SIZE
    inputs[dest+4]=clamp(inputs[dest+4]!,-2,2)
    inputs[dest+5]=clamp(inputs[dest+5]!,-2,2)
  }
  return inputs
}
/** Match the CPU/GPU state projection: pins, mouse constraints, distance relaxation,
 * floor and spherical collision are applied on host in BOTH lanes.
 */
export function projectCloth(cloth:Cloth,values:Float32Array,width:number,
  wires:readonly[number,number],interaction:ClothInteraction):Cloth{
  const next=cloneCloth(cloth),count=cloth.columns*cloth.rows
  if(values.length!==count*width)throw new Error('cloth output shape mismatch')
  const state=next.state,original=cloth.state
  for(let i=0;i<count;i++){
    const at=i*4
    const x=values[i*width+wires[0]]!,y=values[i*width+wires[1]]!
    if(!Number.isFinite(x)||!Number.isFinite(y))throw new Error('nonfinite cloth node')
    state[at]=clamp(x,.012,.988);state[at+1]=clamp(y,.012,.988)
    state[at+2]=original[at]!;state[at+3]=original[at+1]!
  }
  const lock=(i:number)=>{
    if(i===interaction.dragged){state[i*4]=interaction.x;state[i*4+1]=interaction.y;return true}
    if(next.pinned[i]===1){
      state[i*4]=next.origin[i*2]!;state[i*4+1]=next.origin[i*2+1]!
      return true
    }
    return false
  }
  const held=new Uint8Array(count)
  for(let i=0;i<count;i++)if(lock(i))held[i]=1
  for(let iter=0;iter<3;iter++){
    for(const edge of next.springs){
      if(!edge.alive)continue
      const a=edge.a*4,b=edge.b*4
      const dx=state[b]!-state[a]!,dy=state[b+1]!-state[a+1]!
      const distance=Math.hypot(dx,dy)
      if(distance<1e-8)continue
      const correction=clamp((distance-edge.rest)/distance,-.65,.65)*.36
      if(held[edge.a]===0){state[a]+=dx*correction*(held[edge.b]===1?2:1);state[a+1]+=dy*correction*(held[edge.b]===1?2:1)}
      if(held[edge.b]===0){state[b]-=dx*correction*(held[edge.a]===1?2:1);state[b+1]-=dy*correction*(held[edge.a]===1?2:1)}
    }
    for(let i=0;i<count;i++){
      if(held[i]===1)continue
      const at=i*4,x=state[at]!-.5,y=state[at+1]!-.78
      const radius=Math.hypot(x,y),sphere=.14
      if(radius<sphere&&radius>1e-8) {
        state[at]=.5+x*sphere/radius;state[at+1]=.78+y*sphere/radius
      }
      state[at]=clamp(state[at]!,.008,.992)
      state[at+1]=clamp(state[at+1]!,.008,.955)
    }
  }
  next.step++
  return next
}
export function tearNear(cloth:Cloth,x:number,y:number,radius=.042):number{
  let removed=0
  for(const spring of cloth.springs){
    if(!spring.alive)continue
    const ax=cloth.state[spring.a*4]!,ay=cloth.state[spring.a*4+1]!
    const bx=cloth.state[spring.b*4]!,by=cloth.state[spring.b*4+1]!
    const dx=bx-ax,dy=by-ay,t=clamp(((x-ax)*dx+(y-ay)*dy)/Math.max(1e-9,dx*dx+dy*dy),0,1)
    if(Math.hypot(ax+dx*t-x,ay+dy*t-y)<radius){spring.alive=false;removed++}
  }
  return removed
}
export function nearestNode(cloth:Cloth,x:number,y:number):number|null{
  let winner=-1,best=.05*.05
  for(let i=0;i<cloth.columns*cloth.rows;i++){
    const dx=cloth.state[i*4]!-x,dy=cloth.state[i*4+1]!-y,d=dx*dx+dy*dy
    if(d<best){best=d;winner=i}
  }
  return winner>=0?winner:null
}
export function cpuClothFrame(cloth:Cloth,interaction:ClothInteraction){
  const {circuit,outputWires}=buildClothCircuit()
  const sources=clothSources(cloth,interaction),width=circuit.sourceSize+circuit.operations.length
  const values=new Float32Array(cloth.columns*cloth.rows*width)
  for(let i=0;i<cloth.columns*cloth.rows;i++)
    values.set(circuit.evaluateReference(sources.subarray(i*CLOTH_SOURCE_SIZE,(i+1)*CLOTH_SOURCE_SIZE)),i*width)
  return {next:projectCloth(cloth,values,width,outputWires,interaction),sources,values}
}

/** Physically motivated cohesive-spring fracture reference.
 *
 * Hookean axial forces, Kelvin-Voigt damping and irreversible strain damage.
 * This is a discrete structural graph, NOT an FEM continuum, real-world
 * material certification or volumetric crack propagation solver.
 */
export type XYZ=readonly [number,number,number]
export type BondedBody={
 positions:Float32Array // vec4 x,y,z,inverseMass; inverseMass=0 means anchored
 velocities:Float32Array // vec4 vx,vy,vz,unused
 endpoints:Uint32Array // [nodeA,nodeB] for each edge
 bonds:Float32Array // rest, stiffness, damping, yieldStrain, criticalStrain, damageRate, damage, active
}
export type BondedSettings={dt:number,gravity:XYZ}
export const structuralDefaults:BondedSettings={dt:1/120,gravity:[0,-9.81,0]}
export type FractureDiagnostics={nodes:number,bonds:number,intact:number,
 broken:number,events:number,peakStrain:number,kinetic:number,elastic:number,
 momentum:XYZ}
const stride=8
export function validateBonded(body:BondedBody){
 const n=body.positions.length/4,m=body.endpoints.length/2
 if(!Number.isInteger(n)||!Number.isInteger(m)||n<2||n>16_384||
   m<1||m>131_072||body.positions.length%4!==0||
   body.velocities.length!==body.positions.length||
   body.endpoints.length%2!==0||body.bonds.length!==m*stride)
   throw new Error('invalid bonded body layout or supported particle budget')
 for(let i=0;i<body.positions.length;i++){
  if(!Number.isFinite(body.positions[i]!)||!Number.isFinite(body.velocities[i]!))
   throw new Error('nonfinite node state')
  if(i%4===3&&body.positions[i]!<0)throw new Error('negative inverse mass')
 }
 for(let i=0;i<m;i++){
  const a=body.endpoints[i*2]!,b=body.endpoints[i*2+1]!,k=i*stride
  if(a>=n||b>=n||a===b)throw new Error('invalid bond endpoints')
  const values=Array.from(body.bonds.subarray(k,k+stride))
  if(values.some(v=>!Number.isFinite(v)))throw new Error('nonfinite bond property')
  const [rest,elastic,damping,yieldStrain,critical,rate,damage,active]=values
  if(rest!<=0||elastic!<0||damping!<0||yieldStrain!<0||critical!<=yieldStrain!||
    rate!<0||damage!<0||damage!>1||(active!==0&&active!==1))
    throw new Error('invalid spring or material damage definition')
 }
 return {nodes:n,bonds:m}
}
export function validateStructuralSettings(p:BondedSettings){
 if(!Number.isFinite(p.dt)||p.dt<=0||p.dt>.05||
  !p.gravity.every(v=>Number.isFinite(v)&&Math.abs(v)<=200))
  throw new Error('invalid structural time step/gravity')
}
export function structuralDiagnostics(body:BondedBody):FractureDiagnostics{
 const {nodes,bonds}=validateBonded(body)
 const momentum:[number,number,number]=[0,0,0]
 let intact=0,kinetic=0,elastic=0,peakStrain=0
 for(let i=0;i<nodes;i++){
  const base=i*4,inv=body.positions[base+3]!
  if(inv===0)continue
  const mass=1/inv
  for(let k=0;k<3;k++)momentum[k]=momentum[k]!+mass*body.velocities[base+k]!
  kinetic+=.5*mass*(body.velocities[base]!**2+body.velocities[base+1]!**2+
    body.velocities[base+2]!**2)
 }
 for(let i=0;i<bonds;i++){
  const start=i*stride
  if(body.bonds[start+7]!==1)continue
  intact++
  const a=body.endpoints[i*2]!*4,b=body.endpoints[i*2+1]!*4
  const dist=Math.hypot(body.positions[b]!-body.positions[a]!,
   body.positions[b+1]!-body.positions[a+1]!,
   body.positions[b+2]!-body.positions[a+2]!)
  const extension=dist-body.bonds[start]!
  peakStrain=Math.max(peakStrain,Math.abs(extension/body.bonds[start]!))
  elastic+=.5*body.bonds[start+1]!*extension*extension
 }
 return {nodes,bonds,intact,broken:bonds-intact,events:0,peakStrain,
  kinetic,elastic,momentum}
}
export function applyStructuralImpulse(body:BondedBody,node:number,impulse:XYZ):BondedBody{
 const {nodes}=validateBonded(body)
 if(!Number.isSafeInteger(node)||node<0||node>=nodes||!impulse.every(Number.isFinite))
  throw new Error('invalid impact target or impulse')
 const state={...body,velocities:body.velocities.slice()}
 const inv=body.positions[node*4+3]!
 for(let k=0;k<3;k++)state.velocities[node*4+k]=Math.fround(
  state.velocities[node*4+k]!+impulse[k]!*inv)
 return state
}
export function stepBondedCpu(initial:BondedBody,
  options:BondedSettings=structuralDefaults){
 const {nodes,bonds}=validateBonded(initial)
 validateStructuralSettings(options)
 const body:BondedBody={
  positions:initial.positions.slice(),velocities:initial.velocities.slice(),
  endpoints:initial.endpoints,bonds:initial.bonds.slice()
 }
 const forces=new Float64Array(nodes*3)
 let events=0,peakStrain=0
 for(let i=0;i<bonds;i++){
  const start=i*stride
  if(body.bonds[start+7]===0)continue
  const a=body.endpoints[2*i]!*4,b=body.endpoints[2*i+1]!*4
  const dx=body.positions[b]!-body.positions[a]!
  const dy=body.positions[b+1]!-body.positions[a+1]!
  const dz=body.positions[b+2]!-body.positions[a+2]!
  const dist=Math.hypot(dx,dy,dz)
  const rest=body.bonds[start]!
  // Coincident nodes cannot establish a unique axial orientation. Skip
  // force but permit damage at this collapsed configuration.
  const strain=Math.abs(dist-rest)/rest
  peakStrain=Math.max(peakStrain,strain)
  const yieldStrain=body.bonds[start+3]!,critical=body.bonds[start+4]!
  if(strain>yieldStrain){
   const rate=body.bonds[start+5]!
   const increment=options.dt*rate*(strain-yieldStrain)/
     Math.max(1e-8,critical-yieldStrain)
   body.bonds[start+6]=Math.fround(Math.min(1,body.bonds[start+6]!+increment))
  }
  if(strain>=critical||body.bonds[start+6]!>=1){
   body.bonds[start+7]=0
   events++
   continue
  }
  if(dist<=1e-12)continue
  const nx=dx/dist,ny=dy/dist,nz=dz/dist
  const rel=(body.velocities[b]!-body.velocities[a]!)*nx+
   (body.velocities[b+1]!-body.velocities[a+1]!)*ny+
   (body.velocities[b+2]!-body.velocities[a+2]!)*nz
  const magnitude=body.bonds[start+1]!*(dist-rest)+body.bonds[start+2]!*rel
  const aa=(a/4)*3,bb=(b/4)*3
  forces[aa]=forces[aa]!+magnitude*nx
  forces[aa+1]=forces[aa+1]!+magnitude*ny
  forces[aa+2]=forces[aa+2]!+magnitude*nz
  forces[bb]=forces[bb]!-magnitude*nx
  forces[bb+1]=forces[bb+1]!-magnitude*ny
  forces[bb+2]=forces[bb+2]!-magnitude*nz
 }
 for(let i=0;i<nodes;i++){
  const base=i*4,force=i*3,inv=body.positions[base+3]!
  if(inv===0)continue
  for(let k=0;k<3;k++){
   body.velocities[base+k]=Math.fround(body.velocities[base+k]!+
      options.dt*(options.gravity[k]!+forces[force+k]!*inv))
   body.positions[base+k]=Math.fround(body.positions[base+k]!+
      options.dt*body.velocities[base+k]!)
  }
 }
 const diagnostics=structuralDiagnostics(body)
 return {body,diagnostics:{...diagnostics,events,peakStrain}}
}
/** Procedural discretized 3D bridge with diagonal tensile members.
 * Structural values are dimensionless demonstration parameters, not
 * engineering-certified steel/concrete properties.
 */
export function makeBondedBridge(columns=14,levels=4,spacing=.65):BondedBody{
 if(!Number.isInteger(columns)||columns<4||columns>64||
    !Number.isInteger(levels)||levels<2||levels>10||
    !Number.isFinite(spacing)||spacing<=0)throw new Error('bad bridge mesh resolution')
 const pos:number[]=[],vel:number[]=[],edges:number[]=[],bonds:number[]=[]
 const node=(x:number,y:number,z:number,anchor=false)=>{
  const index=pos.length/4
  pos.push(x,y,z,anchor?0:1)
  vel.push(0,0,0,0)
  return index
 }
 const ids:number[][][]=[]
 for(let x=0;x<columns;x++){
  const slice:number[][]=[]
  for(let y=0;y<levels;y++){
   const row:number[]=[]
   for(let z=0;z<2;z++)
    row.push(node((x-(columns-1)/2)*spacing,
      1.4+y*spacing,(z-.5)*spacing,
      (x===0||x===columns-1)&&y===0))
   slice.push(row)
  }
  ids.push(slice)
 }
 const bind=(i:number,j:number)=>{
  const a=i*4,b=j*4,rest=Math.hypot(pos[b]!-pos[a]!,pos[b+1]!-pos[a+1]!,pos[b+2]!-pos[a+2]!)
  edges.push(i,j)
  bonds.push(Math.fround(rest),72,1.7,.11,.39,3.5,0,1)
 }
 for(let x=0;x<columns;x++)for(let y=0;y<levels;y++)for(let z=0;z<2;z++){
  const i=ids[x]![y]![z]!
  if(x+1<columns)bind(i,ids[x+1]![y]![z]!)
  if(y+1<levels)bind(i,ids[x]![y+1]![z]!)
  if(z===0)bind(i,ids[x]![y]![1]!)
  if(x+1<columns&&y+1<levels)bind(i,ids[x+1]![y+1]![z]!)
  if(x+1<columns&&z===0)bind(i,ids[x+1]![y]![1]!)
 }
 return {positions:Float32Array.from(pos),velocities:Float32Array.from(vel),
  endpoints:Uint32Array.from(edges),bonds:Float32Array.from(bonds)}
}

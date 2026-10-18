import {Circuit} from '../../../src/runtime/numeric-ui-core.ts'

/**
 * Independent CPU reference geometry and Monte Carlo light-transport integrator.
 * Synthetic Cornell-inspired test room. No pretrained textures, models or images.
 * Limitations: sphere/axis-plane primitive traversal; BVH, GGX microfacets and
 * spectral color are future milestones. This is actual path tracing, not animation.
 */
export type Vec3=readonly [number,number,number]
export type Ray={origin:Vec3,direction:Vec3}
export type Material='diffuse'|'mirror'|'dielectric'|'emissive'
export type Hit={t:number,position:Vec3,normal:Vec3,color:Vec3,
  material:Material,emission:Vec3,frontFace:boolean}
export type Sphere={center:Vec3,radius:number,color:Vec3,material:Material}
export const EPSILON=.0008
export const MAX_BOUNCES=5
export const LIGHT_POWER:Vec3=[18,16,13]
export const LIGHT_AREA=.9
const PI=Math.PI
export const add=(a:Vec3,b:Vec3):Vec3=>[a[0]+b[0],a[1]+b[1],a[2]+b[2]]
export const sub=(a:Vec3,b:Vec3):Vec3=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]]
export const scale=(a:Vec3,k:number):Vec3=>[a[0]*k,a[1]*k,a[2]*k]
export const mul=(a:Vec3,b:Vec3):Vec3=>[a[0]*b[0],a[1]*b[1],a[2]*b[2]]
export const dot=(a:Vec3,b:Vec3)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
export const cross=(a:Vec3,b:Vec3):Vec3=>[
  a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
export const length=(v:Vec3)=>Math.sqrt(dot(v,v))
export const unit=(v:Vec3):Vec3=>{
  const n=length(v)
  if(n<1e-15||!Number.isFinite(n))throw new Error('invalid zero-length ray direction')
  return scale(v,1/n)
}
export const reflect=(d:Vec3,n:Vec3):Vec3=>sub(d,scale(n,2*dot(d,n)))
export function refract(d:Vec3,n:Vec3,etaRatio:number):Vec3|null{
  const c=Math.min(1,-dot(d,n))
  const perpendicular=scale(add(d,scale(n,c)),etaRatio)
  const parallelSquared=1-dot(perpendicular,perpendicular)
  if(parallelSquared<0)return null
  return unit(add(perpendicular,scale(n,-Math.sqrt(parallelSquared))))
}
export function schlick(cosine:number,etaRatio:number):number{
  const r0=((1-etaRatio)/(1+etaRatio))**2
  return r0+(1-r0)*(1-Math.max(0,Math.min(1,cosine)))**5
}
export function intersectSphere(ray:Ray,sphere:Sphere,tMin=EPSILON,tMax=Infinity):Hit|null{
  if(!Number.isFinite(sphere.radius)||sphere.radius<=0)throw new Error('invalid sphere radius')
  const offset=sub(ray.origin,sphere.center)
  const a=dot(ray.direction,ray.direction)
  const halfB=dot(offset,ray.direction)
  const discriminant=halfB*halfB-a*(dot(offset,offset)-sphere.radius*sphere.radius)
  if(discriminant<0)return null
  const root=Math.sqrt(discriminant)
  let t=(-halfB-root)/a
  if(t<tMin||t>=tMax){
    t=(-halfB+root)/a
    if(t<tMin||t>=tMax)return null
  }
  const position=add(ray.origin,scale(ray.direction,t))
  const outward=scale(sub(position,sphere.center),1/sphere.radius)
  const frontFace=dot(ray.direction,outward)<0
  return {t,position,normal:frontFace?outward:scale(outward,-1),
    color:sphere.color,material:sphere.material,emission:[0,0,0],frontFace}
}
export const spheres:readonly Sphere[]=[
  {center:[-.70,-.61,-2.92],radius:.74,color:[.96,.97,.99],material:'mirror'},
  {center:[.74,-.74,-3.18],radius:.59,color:[.99,.99,1],material:'dielectric'},
  {center:[.20,-1.08,-1.89],radius:.27,color:[.92,.62,.30],material:'diffuse'},
]
type Wall={axis:0|1|2,position:number,normal:Vec3,color:Vec3,material:Material}
const walls:readonly Wall[]=[
  {axis:0,position:-1.72,normal:[1,0,0],color:[.80,.12,.13],material:'diffuse'},
  {axis:0,position:1.72,normal:[-1,0,0],color:[.12,.70,.25],material:'diffuse'},
  {axis:1,position:-1.37,normal:[0,1,0],color:[.84,.82,.80],material:'diffuse'},
  {axis:1,position:1.37,normal:[0,-1,0],color:[.85,.85,.83],material:'diffuse'},
  {axis:2,position:-4.66,normal:[0,0,1],color:[.79,.81,.87],material:'diffuse'},
]
function planeHit(ray:Ray,wall:Wall,tMin:number,tMax:number):Hit|null{
  const axis=wall.axis,d=ray.direction[axis]
  if(Math.abs(d)<1e-8)return null
  const t=(wall.position-ray.origin[axis])/d
  if(t<tMin||t>=tMax)return null
  const p=add(ray.origin,scale(ray.direction,t))
  if(Math.abs(p[0])>1.72+EPSILON||Math.abs(p[1])>1.37+EPSILON||
    p[2]<-4.66-EPSILON||p[2]>1.2)return null
  const frontFace=dot(ray.direction,wall.normal)<0
  return {t,position:p,normal:frontFace?wall.normal:scale(wall.normal,-1),
    color:wall.color,material:wall.material,emission:[0,0,0],frontFace}
}
function lightHit(ray:Ray,tMin:number,tMax:number):Hit|null{
  const d=ray.direction[1]
  if(Math.abs(d)<1e-9)return null
  const t=(1.365-ray.origin[1])/d
  if(t<tMin||t>=tMax)return null
  const p=add(ray.origin,scale(ray.direction,t))
  if(p[0]<-.45||p[0]>.45||p[2]<-3.35||p[2]>-2.35)return null
  const outward:Vec3=[0,-1,0]
  const frontFace=dot(ray.direction,outward)<0
  return {t,position:p,normal:outward,color:[1,1,1],material:'emissive',
    emission:frontFace?LIGHT_POWER:[0,0,0],frontFace}
}
export function intersectScene(ray:Ray,tMin=EPSILON,tMax=Infinity):Hit|null{
  let best:Hit|null=null
  for(const sphere of spheres){
    const hit=intersectSphere(ray,sphere,tMin,best?.t??tMax)
    if(hit)best=hit
  }
  const light=lightHit(ray,tMin,best?.t??tMax)
  if(light)best=light
  for(const wall of walls){
    const hit=planeHit(ray,wall,tMin,best?.t??tMax)
    if(hit)best=hit
  }
  return best
}
export function random(seed:number){
  let state=(seed>>>0)||1
  return()=>{
    state^=state<<13;state^=state>>>17;state^=state<<5
    return (state>>>0)/4294967296
  }
}
export function cosineHemisphere(normal:Vec3,u:number,v:number):Vec3{
  const radius=Math.sqrt(Math.max(0,Math.min(1,u))),phi=2*PI*v
  const w=Math.sqrt(Math.max(0,1-u))
  const tangent=unit(cross(Math.abs(normal[1])<.9?[0,1,0]:[1,0,0],normal))
  const bitangent=cross(normal,tangent)
  return unit(add(add(scale(tangent,radius*Math.cos(phi)),
    scale(bitangent,radius*Math.sin(phi))),scale(normal,w)))
}
export function buildLightThroughputCircuit(){
  const circuit=new Circuit(6)
  const sources=Array.from({length:6},(_,index)=>circuit.source(index))
  const outputWires=[0,1,2].map(i=>circuit.multiply(sources[i]!,sources[i+3]!))
  return {circuit,outputWires}
}
export function sampleAreaLight(random01:()=>number):Vec3{
  return [(random01()-.5)*.9,1.365,-3.35+random01()]
}
export function estimateDirectLighting(hit:Hit,random01:()=>number):Vec3{
  const point=sampleAreaLight(random01)
  const direction=sub(point,hit.position)
  const distance2=dot(direction,direction)
  if(distance2<1e-9)return [0,0,0]
  const distance=Math.sqrt(distance2)
  const wi=scale(direction,1/distance)
  const cosSurface=Math.max(0,dot(hit.normal,wi))
  const cosLight=Math.max(0,wi[1])
  if(cosSurface<=0||cosLight<=0)return [0,0,0]
  const visibility=intersectScene({origin:add(hit.position,scale(hit.normal,EPSILON*2)),direction:wi},
    EPSILON,Math.max(EPSILON,distance-EPSILON*4))
  if(visibility)return [0,0,0]
  const f=LIGHT_AREA*cosSurface*cosLight/(PI*distance2)
  return mul(hit.color,scale(LIGHT_POWER,f))
}
/** Reference unbiased diffuse light transport with next-event estimation,
 * delta mirror/glass transport and simple Russian roulette. No MIS claims.
 */
export function tracePathCpu(ray:Ray,seed:number,maxBounces=MAX_BOUNCES):Vec3{
  if(!Number.isSafeInteger(maxBounces)||maxBounces<1||maxBounces>16)
    throw new Error('invalid path-tracing bounce count')
  const rand=random(seed)
  let current:Ray={origin:ray.origin,direction:unit(ray.direction)}
  let throughput:Vec3=[1,1,1],radiance:Vec3=[0,0,0],specular=true
  for(let bounce=0;bounce<maxBounces;bounce++){
    const hit=intersectScene(current)
    if(!hit)break
    if(hit.material==='emissive'){
      if(specular)radiance=add(radiance,mul(throughput,hit.emission))
      break
    }
    if(hit.material==='diffuse'){
      radiance=add(radiance,mul(throughput,estimateDirectLighting(hit,rand)))
      throughput=mul(throughput,hit.color)
      current={origin:add(hit.position,scale(hit.normal,EPSILON*3)),
        direction:cosineHemisphere(hit.normal,rand(),rand())}
      specular=false
    }else if(hit.material==='mirror'){
      throughput=mul(throughput,hit.color)
      const direction=unit(reflect(current.direction,hit.normal))
      current={origin:add(hit.position,scale(direction,EPSILON*3)),direction}
      specular=true
    }else{
      const eta=hit.frontFace?1/1.5:1.5
      const c=Math.min(1,-dot(current.direction,hit.normal))
      const transmitted=refract(current.direction,hit.normal,eta)
      const direction=transmitted===null||rand()<schlick(c,eta)
        ? unit(reflect(current.direction,hit.normal)):transmitted
      throughput=mul(throughput,hit.color)
      current={origin:add(hit.position,scale(direction,EPSILON*3)),direction}
      specular=true
    }
    if(bounce>=3){
      const survival=Math.min(.98,Math.max(.05,...throughput))
      if(rand()>survival)break
      throughput=scale(throughput,1/survival)
    }
  }
  return radiance
}
export function cameraRay(px:number,py:number,width:number,height:number,
  yaw=0,pitch=0):Ray{
  if(![px,py,width,height,yaw,pitch].every(Number.isFinite)||
    width<=0||height<=0||px<0||py<0||px>=width||py>=height)
    throw new Error('invalid camera sample')
  const x=(2*(px+.5)/width-1)*(width/height)*Math.tan(48*PI/360)
  const y=(1-2*(py+.5)/height)*Math.tan(48*PI/360)
  const z=-1
  const c=Math.cos(yaw),s=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch)
  return {origin:[0,0,2.4],
    direction:unit([c*x+s*z,cp*y-sp*(c*z-s*x),sp*y+cp*(c*z-s*x)])}
}
export function renderCpuPixel(x:number,y:number,width:number,height:number,
  samples=16,seed=123):Vec3{
  if(!Number.isSafeInteger(samples)||samples<1||samples>8192)
    throw new Error('invalid CPU ray sample count')
  let sum:Vec3=[0,0,0]
  for(let i=0;i<samples;i++)sum=add(sum,
    tracePathCpu(cameraRay(x,y,width,height),seed+i*9973))
  return scale(sum,1/samples)
}

/** Procedural instanced 3D linkages draw from current GPU joint state.
 * All animation comes from physical GPU integration, not scripted angles.
 */
export const legPopulationRenderWgsl=`
struct View {
 right:vec4<f32>,up:vec4<f32>,forward:vec4<f32>,
 eye:vec4<f32>,projection:vec4<f32>,
 gridLayout:vec4<f32>,geometry:vec4<f32>,
};
@group(0) @binding(0) var<storage,read> joints:array<vec4<f32>>;
@group(0) @binding(1) var<uniform> view:View;
struct Out {
 @builtin(position) clip:vec4<f32>,
 @location(0) normal:vec3<f32>,
 @location(1) tint:vec3<f32>,
 @location(2) world:vec3<f32>,
};
const PI:f32=3.141592653589793;
fn root(i:u32)->vec3<f32>{
 let width=u32(view.gridLayout.x);
 let column=f32(i%width);
 let row=f32(i/width);
 return vec3<f32>(
  (column-(view.gridLayout.x-1.0)*.5)*view.gridLayout.z,
  view.geometry.z,
  (row-(view.gridLayout.y-1.0)*.5)*view.gridLayout.z);
}
fn nodes(i:u32)->array<vec3<f32>,3>{
 let state=joints[i];
 let hip=root(i);
 let knee=hip+vec3<f32>(view.geometry.x*sin(state.x),
  -view.geometry.x*cos(state.x),0.0);
 let foot=knee+vec3<f32>(view.geometry.y*sin(state.x+state.y),
  -view.geometry.y*cos(state.x+state.y),0.0);
 return array<vec3<f32>,3>(hip,knee,foot);
}
fn project(p:vec3<f32>)->vec4<f32>{
 let relative=p-view.eye.xyz;
 let depth=dot(relative,view.forward.xyz);
 return vec4<f32>(
  dot(relative,view.right.xyz)*view.projection.x,
  dot(relative,view.up.xyz)*view.projection.y,
  depth*view.projection.z-view.projection.w,depth);
}
@vertex fn rod(@builtin(vertex_index) v:u32,
 @builtin(instance_index) instance:u32)->Out{
 let robot=instance/2u;
 let segment=instance%2u;
 let points=nodes(robot);
 let start=points[segment];
 let finish=points[segment+1u];
 let direction=normalize(finish-start);
 let radialX=normalize(cross(direction,vec3<f32>(0.0,0.0,1.0)));
 let radialY=cross(direction,radialX);
 let side=v/6u;
 let corner=v%6u;
 var angleIndex=side;
 var endpoint=0.0;
 if(corner==1u||corner==3u||corner==4u){endpoint=1.0;}
 if(corner==2u||corner==4u||corner==5u){angleIndex=angleIndex+1u;}
 let angle=2.0*PI*f32(angleIndex)/8.0;
 let normal=radialX*cos(angle)+radialY*sin(angle);
 let world=mix(start,finish,endpoint)+normal*.052;
 let state=joints[robot];
 let speed=clamp(length(state.zw)*.15,0.0,1.0);
 let c1=vec3<f32>(.10,.83,.78);
 let c2=vec3<f32>(1.0,.64,.22);
 var out:Out;
 out.clip=project(world);
 out.normal=normal;
 out.tint=mix(select(c1,c2,segment==1u),vec3<f32>(1.0,.20,.33),speed*.45);
 out.world=world;
 return out;
}
@vertex fn joint(@builtin(vertex_index) v:u32,
 @builtin(instance_index) instance:u32)->Out{
 let robot=instance/3u;
 let which=instance%3u;
 let center=nodes(robot)[which];
 let segment=v/6u;
 let corner=v%6u;
 let lat=segment/6u;
 let lon=segment%6u;
 var u=lon;
 var y=lat;
 if(corner==1u||corner==3u||corner==4u){y=y+1u;}
 if(corner==2u||corner==4u||corner==5u){u=u+1u;}
 let theta=PI*f32(y)/6.0;
 let phi=2.0*PI*f32(u)/6.0;
 let normal=vec3<f32>(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));
 let world=center+normal*.105;
 var out:Out;
 out.clip=project(world);
 out.normal=normal;
 out.tint=select(vec3<f32>(.94,.93,1.0),vec3<f32>(.26,.96,.72),which==2u);
 out.world=world;
 return out;
}
@vertex fn ground(@builtin(vertex_index) v:u32)->Out{
 let corners=array<vec2<f32>,6>(
  vec2<f32>(-1.0,-1.0),vec2<f32>(-1.0,1.0),
  vec2<f32>(1.0,-1.0),vec2<f32>(1.0,-1.0),
  vec2<f32>(-1.0,1.0),vec2<f32>(1.0,1.0));
 let q=corners[v];
 let world=vec3<f32>(
  q.x*(view.gridLayout.x*view.gridLayout.z*.53),
  view.gridLayout.w-.01,
  q.y*(view.gridLayout.y*view.gridLayout.z*.53));
 var out:Out;
 out.clip=project(world);
 out.normal=vec3<f32>(0.0,1.0,0.0);
 let checker=(i32(floor(world.x/view.gridLayout.z))+
  i32(floor(world.z/view.gridLayout.z)))&1;
 out.tint=select(vec3<f32>(.038,.094,.15),vec3<f32>(.062,.13,.20),checker==1);
 out.world=world;
 return out;
}
@fragment fn shadeMachine(inp:Out)->@location(0) vec4<f32>{
 let n=normalize(inp.normal);
 let light=normalize(vec3<f32>(-.55,.88,.58));
 let lambert=max(0.0,dot(n,light));
 let eye=normalize(view.eye.xyz-inp.world);
 let rim=pow(1.0-max(0.0,dot(n,eye)),2.7);
 let specular=pow(max(0.0,dot(reflect(-light,n),eye)),50.0);
 let color=inp.tint*(.2+.68*lambert)+
   rim*.22*vec3<f32>(.4,.75,1.0)+specular*.32*vec3<f32>(1.0);
 return vec4<f32>(
  pow(clamp(color,vec3<f32>(0.0),vec3<f32>(1.0)),
   vec3<f32>(1.0/2.2)),1.0);
}
`

/** GPU-resident, lit mesh spheres, driven directly by the contact solver's
 * continuously updated position buffer. No CPU readback to animate geometry.
 */
export const destructionWgsl=`
struct Camera {
  right:vec4<f32>,
  up:vec4<f32>,
  forward:vec4<f32>,
  eye:vec4<f32>,
  projection:vec4<f32>,
};
@group(0) @binding(0) var<storage,read> position:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> materials:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> camera:Camera;
struct Out {
  @builtin(position) clip:vec4<f32>,
  @location(0) normal:vec3<f32>,
  @location(1) color:vec3<f32>,
  @location(2) world:vec3<f32>,
};
const PI:f32=3.141592653589793;
const LAT:u32=12u;
const LON:u32=20u;
fn sphereVertex(index:u32)->vec3<f32>{
  let tri=index/6u;
  let corner=index%6u;
  let row=tri/LON;
  let column=tri%LON;
  var u=column;
  var v=row;
  if(corner==1u||corner==3u||corner==4u){v=v+1u;}
  if(corner==2u||corner==4u||corner==5u){u=u+1u;}
  let theta=PI*f32(v)/f32(LAT);
  let phi=2.0*PI*f32(u)/f32(LON);
  return vec3<f32>(sin(theta)*cos(phi),cos(theta),sin(theta)*sin(phi));
}
@vertex fn vs(@builtin(vertex_index) vertex:u32,
  @builtin(instance_index) instance:u32)->Out{
  let n=sphereVertex(vertex);
  let sphere=position[instance];
  let world=sphere.xyz+n*sphere.w;
  let relative=world-camera.eye.xyz;
  let depth=dot(relative,camera.forward.xyz);
  var out:Out;
  out.clip=vec4<f32>(
    dot(relative,camera.right.xyz)*camera.projection.x,
    dot(relative,camera.up.xyz)*camera.projection.y,
    depth*camera.projection.z-camera.projection.w,
    depth);
  out.normal=n;
  out.color=materials[instance].xyz;
  out.world=world;
  return out;
}
@fragment fn fs(p:Out)->@location(0) vec4<f32>{
  let n=normalize(p.normal);
  let light=normalize(vec3<f32>(-.65,.92,1.1));
  let warm=normalize(vec3<f32>(.9,.55,-.7));
  let diffuse=max(0.0,dot(n,light));
  let rim=pow(1.0-max(0.0,dot(n,normalize(camera.eye.xyz-p.world))),2.5);
  let spec=pow(max(0.0,dot(reflect(-light,n),normalize(camera.eye.xyz-p.world))),45.0);
  let warmTerm=max(0.0,dot(n,warm))*.19;
  let linear=p.color*(.18+.68*diffuse+warmTerm)+vec3<f32>(.42,.77,1.0)*rim*.18+
    vec3<f32>(1.0,.96,.85)*spec*.5;
  return vec4<f32>(pow(clamp(linear,vec3<f32>(0.0),vec3<f32>(1.0)),vec3<f32>(1.0/2.2)),1.0);
}
`

/** Real-time 3D fluid particle rendering reads SPH positions/velocities directly
 * from GPU buffers. Spherical impostors visualize actual simulated particles;
 * they are not the fluid solver, surface reconstruction or refracted water.
 */
export const cataclysmRenderWgsl=`
struct Camera{
 right:vec4<f32>,up:vec4<f32>,forward:vec4<f32>,
 eye:vec4<f32>,projection:vec4<f32>,
};
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(2) var<uniform> camera:Camera;
struct VertexOut{
 @builtin(position) position:vec4<f32>,
 @location(0) corner:vec2<f32>,
 @location(1) velocity:vec3<f32>,
 @location(2) world:vec3<f32>,
};
@vertex fn vs(@builtin(vertex_index) v:u32,@builtin(instance_index) i:u32)->VertexOut{
 var xy=vec2<f32>(-1.0,-1.0);
 let corner=v%6u;
 if(corner==1u||corner==3u||corner==4u){xy.y=1.0;}
 if(corner==2u||corner==4u||corner==5u){xy.x=1.0;}
 let p=positions[i];
 let radius=0.22;
 let world=p.xyz+radius*(camera.right.xyz*xy.x+camera.up.xyz*xy.y);
 let relative=world-camera.eye.xyz;
 let depth=dot(relative,camera.forward.xyz);
 var result:VertexOut;
 result.position=vec4<f32>(
  dot(relative,camera.right.xyz)*camera.projection.x,
  dot(relative,camera.up.xyz)*camera.projection.y,
  depth*camera.projection.z-camera.projection.w,depth);
 result.corner=xy;
 result.velocity=velocities[i].xyz;
 result.world=p.xyz;
 return result;
}
struct FragmentOut{
 @location(0) color:vec4<f32>,
 @builtin(frag_depth) depth:f32
};
@fragment fn fs(input:VertexOut)->FragmentOut{
 let radius2=dot(input.corner,input.corner);
 if(radius2>1.0){discard;}
 let nz=sqrt(max(.0,1.0-radius2));
 let normal=normalize(camera.right.xyz*input.corner.x+
  camera.up.xyz*input.corner.y+camera.forward.xyz*nz*-1.0);
 let light=normalize(vec3<f32>(-.48,.85,.75));
 let view=normalize(camera.eye.xyz-input.world);
 let diffuse=max(0.0,dot(normal,light));
 let specular=pow(max(0.0,dot(reflect(-light,normal),view)),54.0);
 let rim=pow(1.0-max(0.0,dot(normal,view)),2.8);
 let turbulence=clamp(length(input.velocity)*.085,0.0,1.0);
 let base=mix(vec3<f32>(.08,.34,.64),vec3<f32>(.23,.79,.85),.48+turbulence*.36);
 let color=base*(.21+.65*diffuse)+vec3<f32>(.62,.92,1.0)*rim*.37+
  vec3<f32>(1.0,.97,.90)*specular*.80;
 let relative=input.world-camera.eye.xyz;
 let distance=dot(relative,camera.forward.xyz)-nz*.22;
 let ndcDepth=distance*camera.projection.z-camera.projection.w;
 var output:FragmentOut;
 output.color=vec4<f32>(pow(clamp(color,vec3<f32>(0.0),vec3<f32>(1.0)),
  vec3<f32>(1.0/2.2)),1.0);
 output.depth=clamp(ndcDepth/max(distance,1.0e-4),0.0,1.0);
 return output;
}
`

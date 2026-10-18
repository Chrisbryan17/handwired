"""Execute actual Handwired cohesive-fracture WGSL on NVIDIA Tesla T4 Vulkan.
CPU oracle is independently authored TypeScript. No browser-GPU claim.
"""
import json,sys,os,time,math,traceback,hashlib,struct
from pathlib import Path
import numpy as np
import wgpu
out=Path(sys.argv[1] if len(sys.argv)>1 else '/kaggle/working')
report={'ok':False,'backend':'wgpu-native/Vulkan','browserWebGpuVerified':False,
 'model':'3D cohesive Hookean bonded graph with irreversible failure',
 'cases':[],'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
try:
 os.environ['WGPU_BACKEND_TYPE']='Vulkan'
 adapters=wgpu.gpu.enumerate_adapters_sync()
 report['adapters']=[str(a.info) for a in adapters]
 chosen=next((a for a in adapters if 'tesla t4' in str(a.info).lower()
    and "backend_type='Vulkan'" in str(a.info)
    and 'software' not in str(a.info).lower()),None)
 if chosen is None:raise RuntimeError('No physical Tesla T4 native Vulkan compute adapter')
 report['executingAdapter']=str(chosen.info)
 device=chosen.request_device_sync()
 B=wgpu.BufferUsage
 modules={}
 for name in ('fracture-bond-forces','fracture-handwired-node'):
  code=(out/(name+'.wgsl')).read_text()
  report.setdefault('wgslSha256',{})[name]=hashlib.sha256(code.encode()).hexdigest()
  modules[name]=device.create_shader_module(code=code,label='Handwired '+name)
 bond_pipeline=device.create_compute_pipeline(layout='auto',
   compute={'module':modules['fracture-bond-forces'],'entry_point':'bondUpdate'})
 node_pipeline=device.create_compute_pipeline(layout='auto',
   compute={'module':modules['fracture-handwired-node'],'entry_point':'advanceNodes'})
 def upload(arr,label,extra=0):
  return device.create_buffer_with_data(data=arr.tobytes(),
     usage=B.STORAGE|B.COPY_DST|extra,label=label)
 def alloc(bytes,label,extra=0):
  return device.create_buffer(size=bytes,usage=B.STORAGE|extra,label=label)
 def bind(pipeline,items):
  return device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
   entries=[{'binding':i,'resource':{'buffer':buffer}}
       for i,buffer in enumerate(items)])
 def dispatch(encoder,pipeline,bgroup,count):
  compute=encoder.begin_compute_pass()
  compute.set_pipeline(pipeline)
  compute.set_bind_group(0,bgroup)
  compute.dispatch_workgroups(math.ceil(count/128))
  compute.end()
 reference=json.loads((out/'fracture-native-reference.json').read_text())
 threshold=reference['tolerance']
 for case in reference['scenes']:
  start=time.perf_counter()
  initial=case['initial']
  positions=np.asarray(initial['positions'],dtype=np.float32)
  velocities=np.asarray(initial['velocities'],dtype=np.float32)
  endpoints=np.asarray(initial['endpoints'],dtype=np.uint32)
  bonds=np.asarray(initial['bonds'],dtype=np.float32)
  n=len(positions)//4;m=len(endpoints)//2
  assert n>1 and m>0 and len(bonds)==m*8 and len(velocities)==n*4
  pair=endpoints.reshape(m,2)
  degrees=np.zeros(n,dtype=np.uint32)
  for a,b in pair:
   degrees[a]+=1;degrees[b]+=1
  offsets=np.zeros(n+1,dtype=np.uint32)
  offsets[1:]=np.cumsum(degrees,dtype=np.uint32)
  cursors=offsets[:-1].copy()
  adjacency=np.zeros((m*2,2),dtype=np.uint32)
  for i,(a,b) in enumerate(pair):
   left=int(cursors[a]);right=int(cursors[b])
   adjacency[left]=[i,0]
   adjacency[right]=[i,1]
   cursors[a]+=1;cursors[b]+=1
  material=bonds.reshape(m,8)[:,:4].copy().astype(np.float32)
  state=bonds.reshape(m,8)[:,4:].copy().astype(np.float32)
  settings=case['settings']
  uniforms=bytearray(32)
  struct.pack_into('<2I',uniforms,0,n,m)
  struct.pack_into('<4f',uniforms,16,settings['dt'],*settings['gravity'])
  pos=upload(positions,'GPU structural nodes',B.COPY_SRC)
  vel=upload(velocities,'GPU structural momentum',B.COPY_SRC)
  ends=upload(endpoints,'GPU cohesive graph edges')
  props=upload(material,'GPU spring elastic material')
  damage=upload(state,'GPU permanent strain damage',B.COPY_SRC)
  forces=alloc(m*16,'GPU axial force per bond')
  out_pos=alloc(n*16,'GPU integrated nodes',B.COPY_SRC)
  out_vel=alloc(n*16,'GPU integrated momentum',B.COPY_SRC)
  csr=upload(offsets,'CSR node offsets')
  neighbors=upload(adjacency,'CSR oriented adjacency')
  uniform=device.create_buffer_with_data(data=bytes(uniforms),
      usage=B.UNIFORM|B.COPY_DST,label='Structural timestep')
  bond_bg=bind(bond_pipeline,[pos,vel,ends,props,damage,forces,uniform])
  node_bg=bind(node_pipeline,[pos,vel,csr,neighbors,forces,out_pos,out_vel,uniform])
  for i in range(case['steps']):
   encoder=device.create_command_encoder()
   dispatch(encoder,bond_pipeline,bond_bg,m)
   dispatch(encoder,node_pipeline,node_bg,n)
   encoder.copy_buffer_to_buffer(out_pos,0,pos,0,n*16)
   encoder.copy_buffer_to_buffer(out_vel,0,vel,0,n*16)
   device.queue.submit([encoder.finish()])
  actual_pos=np.frombuffer(device.queue.read_buffer(pos),dtype=np.float32).copy()
  actual_vel=np.frombuffer(device.queue.read_buffer(vel),dtype=np.float32).copy()
  actual_damage=np.frombuffer(device.queue.read_buffer(damage),dtype=np.float32).copy().reshape(m,4)
  expected=case['expected']
  expected_pos=np.asarray(expected['positions'],dtype=np.float32)
  expected_vel=np.asarray(expected['velocities'],dtype=np.float32)
  expected_bond=np.asarray(expected['bonds'],dtype=np.float32).reshape(m,8)
  errors={'positions':float(np.max(np.abs(actual_pos-expected_pos))),
    'velocities':float(np.max(np.abs(actual_vel-expected_vel))),
    'damage':float(np.max(np.abs(actual_damage[:,2]-expected_bond[:,6])))}
  active_mismatch=int(np.count_nonzero(actual_damage[:,3]!=expected_bond[:,7]))
  invalid=int(np.count_nonzero(~np.isfinite(actual_pos))+
      np.count_nonzero(~np.isfinite(actual_vel))+
      np.count_nonzero(~np.isfinite(actual_damage)))
  broken=int(np.count_nonzero(actual_damage[:,3]==0))
  passed=invalid==0 and active_mismatch==0 and all(
      errors[key]<=threshold[key] for key in threshold)
  if case['name']=='true-critical-strain-fracture' and broken==0:
   passed=False
  result={'name':case['name'],'nodes':n,'bonds':m,'steps':case['steps'],
   'errors':errors,'activeMismatches':active_mismatch,'brokenBonds':broken,
   'referenceBroken':case['referenceBroken'],'nonfinite':invalid,
   'elapsedMsIncludesSetupAndReadback':round((time.perf_counter()-start)*1000,2),
   'ok':passed}
  report['cases'].append(result)
  print('PHYSICAL FRACTURE GPU',json.dumps(result),flush=True)
 report['ok']=len(report['cases'])==3 and all(x['ok'] for x in report['cases'])
 report['tolerances']=threshold
 report['boundaries']='Discrete spring network, not FEM or material-certified fracture'
except Exception as error:
 report['error']=repr(error)
 report['traceback']=traceback.format_exc()[-6000:]
finally:
 report['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (out/'native-fracture-report.json').write_text(json.dumps(report,indent=2))
 print('NATIVE_STRUCTURAL_GPU_VERIFIED',report['ok'],flush=True)
 if not report['ok']:sys.exit(1)

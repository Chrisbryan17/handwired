"""Verify three actual 3D Handwired SPH WGSL kernels on NVIDIA Vulkan hardware.
Explicit WebGPU pipeline. The CPU oracle is exported by independent model.ts.
This is *native wgpu/Vulkan* GPU, not Chrome browser WebGPU.
"""
import json,os,sys,time,traceback,struct,hashlib,math
from pathlib import Path
import numpy as np
import wgpu

root=Path(sys.argv[1] if len(sys.argv)>1 else '/kaggle/working')
report={'ok':False,'backend':'wgpu-native/Vulkan',
 'browserWebGpuVerified':False,
 'method':'native hardware WGSL executing actual Handwired 3D SPH compute stages',
 'cases':[],'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
try:
 os.environ['WGPU_BACKEND_TYPE']='Vulkan'
 adapters=wgpu.gpu.enumerate_adapters_sync()
 report['visibleAdapters']=[str(a.info) for a in adapters]
 gpu=next((a for a in adapters if
  'tesla t4' in str(a.info).lower() and
  "backend_type='Vulkan'" in str(a.info) and
  not any(x in str(a.info).lower() for x in ('swiftshader','llvmpipe','software'))),None)
 if gpu is None:raise RuntimeError('Physical Tesla T4 Vulkan adapter missing')
 report['executingAdapter']=str(gpu.info)
 device=gpu.request_device_sync()
 B=wgpu.BufferUsage
 shaders={}
 for name in ('spatial-grid','sph-density','sph-integrate'):
  source=(root/(name+'.wgsl')).read_text()
  report.setdefault('shaderSha256',{})[name]=hashlib.sha256(source.encode()).hexdigest()
  shaders[name]=device.create_shader_module(code=source,label='Handwired '+name)
 # Shared grid pipeline layout matches the exact TypeScript frontiers spatial contract.
 types=['read-only-storage','storage','storage','storage','uniform']
 layout=device.create_bind_group_layout(entries=[
  {'binding':i,'visibility':wgpu.ShaderStage.COMPUTE,'buffer':{'type':t}}
  for i,t in enumerate(types)])
 pipeline_layout=device.create_pipeline_layout(bind_group_layouts=[layout])
 def grid_pipeline(entry):
  return device.create_compute_pipeline(layout=pipeline_layout,
   compute={'module':shaders['spatial-grid'],'entry_point':entry})
 grid_clear=grid_pipeline('clearCells')
 grid_assign=grid_pipeline('assignCells')
 dens_pipeline=device.create_compute_pipeline(layout='auto',
  compute={'module':shaders['sph-density'],'entry_point':'density'})
 flow_pipeline=device.create_compute_pipeline(layout='auto',
  compute={'module':shaders['sph-integrate'],'entry_point':'forceAndAdvance'})
 def upload(raw,flags=0,label='input'):
  return device.create_buffer_with_data(data=raw,
   usage=B.STORAGE|B.COPY_DST|flags,label=label)
 def empty(length,flags=0,label='output'):
  return device.create_buffer(size=length,usage=B.STORAGE|flags,label=label)
 def bind(pipeline,buffers):
  return device.create_bind_group(
   layout=pipeline.get_bind_group_layout(0),entries=[
    {'binding':i,'resource':{'buffer':buf}} for i,buf in enumerate(buffers)])
 def run_compute(encoder,pipeline,bgroup,groups):
  p=encoder.begin_compute_pass()
  p.set_pipeline(pipeline)
  p.set_bind_group(0,bgroup)
  p.dispatch_workgroups(groups)
  p.end()
 suite=json.loads((root/'sph-native-reference.json').read_text())
 tolerances=suite['tolerances']
 for entry in suite['data']:
  started=time.perf_counter()
  settings=entry['settings']
  points=np.asarray(entry['initial']['positions'],dtype=np.float32)
  speeds=np.asarray(entry['initial']['velocities'],dtype=np.float32)
  n=len(points)//4
  assert len(points)==len(speeds) and n>0
  # Exact CPU all-pairs comparisons are limited to 1,024. The GPU-only
  # conservation/stability case is separately admitted up to 131,072.
  assert n<= (131_072 if entry.get('stressOnly',False) else 1024)
  h=settings['h']
  cell_size=max(h,.5)
  mins=settings['min']
  maxs=settings['max']
  dims=[math.ceil((maxs[i]-mins[i])/cell_size)+1 for i in range(3)]
  cells=math.prod(dims)
  cells_bytes=bytearray(64)
  struct.pack_into('<3f',cells_bytes,0,*mins)
  struct.pack_into('<3I',cells_bytes,16,*dims)
  struct.pack_into('<3f',cells_bytes,32,cell_size,h,h*h)
  struct.pack_into('<I',cells_bytes,48,n)
  sph_bytes=bytearray(128)
  struct.pack_into('<3f',sph_bytes,0,*mins)
  struct.pack_into('<3f',sph_bytes,16,*maxs)
  struct.pack_into('<3f',sph_bytes,32,*mins)
  struct.pack_into('<3I',sph_bytes,48,*dims)
  struct.pack_into('<4f',sph_bytes,64,h,settings['dt'],settings['restDensity'],settings['stiffness'])
  struct.pack_into('<4f',sph_bytes,80,settings['viscosity'],settings['boundaryRestitution'],settings['gravity'][0],settings['gravity'][1])
  struct.pack_into('<2f',sph_bytes,96,settings['gravity'][2],cell_size)
  struct.pack_into('<I',sph_bytes,112,n)
  position=upload(points.tobytes(),B.COPY_SRC,label='SPH persistent GPU xyz+mass')
  velocity=upload(speeds.tobytes(),B.COPY_SRC,label='SPH persistent GPU velocity')
  heads=empty(cells*4,label='SPH atomic linked cell heads')
  next_cell=empty(n*4,label='SPH next particle in linked cell')
  counts=empty(n*4,label='SPH unused count binding')
  grid_uniform=device.create_buffer_with_data(data=bytes(cells_bytes),usage=B.UNIFORM|B.COPY_DST)
  sph_uniform=device.create_buffer_with_data(data=bytes(sph_bytes),usage=B.UNIFORM|B.COPY_DST)
  density=empty(n*16,B.COPY_SRC,label='SPH GPU density pressure')
  predicted=empty(n*16,B.COPY_SRC,label='SPH GPU next positions')
  motion=empty(n*16,B.COPY_SRC,label='SPH GPU next velocity')
  grid_bg=device.create_bind_group(layout=layout,entries=[
   {'binding':i,'resource':{'buffer':buf}} for i,buf in enumerate([
    position,heads,next_cell,counts,grid_uniform])])
  dens_bg=bind(dens_pipeline,[position,heads,next_cell,density,sph_uniform])
  flow_bg=bind(flow_pipeline,[position,velocity,heads,next_cell,density,
   predicted,motion,sph_uniform])
  initial_field=None
  for step in range(entry['steps']):
   encoder=device.create_command_encoder()
   # Existing WebGPU algorithm: GPU atomic reset then concurrent linked insertion.
   pass_grid=encoder.begin_compute_pass()
   pass_grid.set_bind_group(0,grid_bg)
   pass_grid.set_pipeline(grid_clear)
   pass_grid.dispatch_workgroups(math.ceil(cells/128))
   pass_grid.set_pipeline(grid_assign)
   pass_grid.dispatch_workgroups(math.ceil(n/128))
   pass_grid.end()
   run_compute(encoder,dens_pipeline,dens_bg,math.ceil(n/128))
   run_compute(encoder,flow_pipeline,flow_bg,math.ceil(n/128))
   encoder.copy_buffer_to_buffer(predicted,0,position,0,n*16)
   encoder.copy_buffer_to_buffer(motion,0,velocity,0,n*16)
   device.queue.submit([encoder.finish()])
   if step==0:
    initial_field=np.frombuffer(device.queue.read_buffer(density),
     dtype=np.float32).copy().reshape(n,4)
  final_pos=np.frombuffer(device.queue.read_buffer(position),
   dtype=np.float32).copy()
  final_vel=np.frombuffer(device.queue.read_buffer(velocity),
   dtype=np.float32).copy()
  if entry.get('stressOnly',False):
   coords=final_pos.reshape(n,4)[:,:3]
   speed=np.linalg.norm(final_vel.reshape(n,4)[:,:3],axis=1)
   nonfinite=int(np.count_nonzero(~np.isfinite(final_pos))+
    np.count_nonzero(~np.isfinite(final_vel))+
    np.count_nonzero(~np.isfinite(initial_field)))
   min_box=np.asarray(settings['min'],dtype=np.float32)
   max_box=np.asarray(settings['max'],dtype=np.float32)
   boundary_violations=int(np.count_nonzero((coords<min_box-1e-4)|(coords>max_box+1e-4)))
   mass_error=abs(float(np.sum(final_pos.reshape(n,4)[:,3],dtype=np.float64))-
    entry['expectedMass'])
   passed=nonfinite==0 and mass_error<=1e-5 and boundary_violations==0 and (
    float(np.max(speed))<250)
   stress={'name':entry['name'],'particles':n,'steps':entry['steps'],
    'mode':'physical GPU conservation/stability stress; not CPU all-pairs equivalence',
    'nonfinite':nonfinite,'boundaryViolations':boundary_violations,
    'massError':mass_error,'peakSpeed':float(np.max(speed)),
    'pressureMaximumInitial':float(np.max(initial_field[:,1])),
    'elapsedWithExplicitDiagnosticReadbacksMs':round(
     (time.perf_counter()-started)*1000,3),'ok':passed}
   report['cases'].append(stress)
   print('SPH NATIVE GPU STRESS',json.dumps(stress),flush=True)
   continue
  target=entry['expected']
  oracle_pos=np.asarray(target['positions'],dtype=np.float32)
  oracle_vel=np.asarray(target['velocities'],dtype=np.float32)
  oracle_density=np.asarray(entry['field']['density'],dtype=np.float32)
  oracle_pressure=np.asarray(entry['field']['pressure'],dtype=np.float32)
  errors={
   'positions':float(np.max(np.abs(final_pos-oracle_pos))),
   'velocities':float(np.max(np.abs(final_vel-oracle_vel))),
   'density':float(np.max(np.abs(initial_field[:,0]-oracle_density))),
   'pressure':float(np.max(np.abs(initial_field[:,1]-oracle_pressure))),
  }
  actual_mass=float(np.sum(final_pos.reshape(n,4)[:,3],dtype=np.float64))
  mass_error=abs(actual_mass-entry['expectedMass'])
  invalid=int(np.count_nonzero(~np.isfinite(final_pos))+
   np.count_nonzero(~np.isfinite(final_vel))+
   np.count_nonzero(~np.isfinite(initial_field)))
  passed=invalid==0 and mass_error<=1e-6 and all(
   errors[name]<=tolerances[name] for name in errors)
  entry_report={'name':entry['name'],'particles':n,'steps':entry['steps'],
   'errors':errors,'nonfinite':invalid,'massError':mass_error,
   'ok':passed,'elapsedWithExplicitDiagnosticReadbacksMs':
       round((time.perf_counter()-started)*1000,3)}
  report['cases'].append(entry_report)
  print('SPH NATIVE',json.dumps(entry_report),flush=True)
 report['ok']=len(report['cases'])==4 and all(c['ok'] for c in report['cases'])
 report['tolerances']=tolerances
 report['scientificLimits']='WCSPH bonded to GPU grid; not coastal tsunami CFD or free-surface reconstruction'
except Exception as error:
 report['error']=repr(error)
 report['traceback']=traceback.format_exc()[-6500:]
finally:
 report['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (root/'handwired-native-sph-report.json').write_text(json.dumps(report,indent=2))
 print('FULL_SPH_NATIVE_HARDWARE_PASS',report['ok'],flush=True)
 if not report['ok']:sys.exit(1)

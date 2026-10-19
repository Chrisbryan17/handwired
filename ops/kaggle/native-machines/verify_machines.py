"""True native physical NVIDIA Vulkan validation for Handwired two-link
articulated leg physics. This is pinned-hip 2DOF, NOT full robot locomotion
or browser WebGPU verification.
"""
import json,sys,os,time,traceback,math,hashlib,struct
from pathlib import Path
import numpy as np
import wgpu
out=Path(sys.argv[1] if len(sys.argv)>1 else '/kaggle/working')
report={'ok':False,'backend':'wgpu-native/Vulkan',
 'browserWebGpuVerified':False,
 'controller':'Real Handwired-generated PD circuit, not a learned policy',
 'physics':'Coupled 2DOF Euler-Lagrange leg inertia + ground penalty contact',
 'results':[],'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
try:
 os.environ['WGPU_BACKEND_TYPE']='Vulkan'
 adapters=wgpu.gpu.enumerate_adapters_sync()
 report['visibleAdapters']=[str(a.info) for a in adapters]
 adapter=next((a for a in adapters if
  'tesla t4' in str(a.info).lower() and
  "backend_type='Vulkan'" in str(a.info) and
  not any(x in str(a.info).lower() for x in ('software','swiftshader','llvmpipe'))),None)
 if adapter is None:raise RuntimeError('NVIDIA Tesla T4 physical Vulkan adapter not found')
 report['executingAdapter']=str(adapter.info)
 device=adapter.request_device_sync()
 code=(out/'machines-articulated.wgsl').read_text()
 report['wgslSha256']=hashlib.sha256(code.encode()).hexdigest()
 shader=device.create_shader_module(code=code,label='Handwired actual GPU articulated leg dynamics')
 pipeline=device.create_compute_pipeline(layout='auto',
   compute={'module':shader,'entry_point':'advanceLegs'})
 reference=json.loads((out/'machines-native-reference.json').read_text())
 B=wgpu.BufferUsage
 for item in reference['scenarios']:
  begin=time.perf_counter()
  n=item['count'];steps=item['steps']
  before=np.asarray(item['initial'],dtype=np.float32)
  expected=np.asarray(item['expected'],dtype=np.float32)
  assert len(before)==n*4 and len(expected)==n*4
  p=item['parameters']
  floats=[
   p['length1'],p['length2'],p['mass1'],p['mass2'],
   p['gravity'],p['hipHeight'],p['groundHeight'],p['dt'],
   p['stiffness'],p['contactDamping'],p['friction'],p['tangentDamping'],
   p['kp'],p['kd'],p['torqueLimit'],p['jointDamping'],
   p['targetHip'],p['targetKnee'],0,0]
  uniform=bytearray(96)
  struct.pack_into('<20f',uniform,0,*floats)
  struct.pack_into('<I',uniform,80,n)
  param_buffer=device.create_buffer_with_data(data=bytes(uniform),
   usage=B.UNIFORM|B.COPY_DST,label='Articulated model physical settings')
  current=device.create_buffer_with_data(data=before.tobytes(),
   usage=B.STORAGE|B.COPY_DST|B.COPY_SRC,label='Articulated joint state A')
  next_buffer=device.create_buffer(size=n*16,
   usage=B.STORAGE|B.COPY_DST|B.COPY_SRC,label='Articulated joint state B')
  def bind(a,b):
   return device.create_bind_group(layout=pipeline.get_bind_group_layout(0),
    entries=[
     {'binding':0,'resource':{'buffer':a}},
     {'binding':1,'resource':{'buffer':b}},
     {'binding':2,'resource':{'buffer':param_buffer}},
    ])
  binds=[bind(current,next_buffer),bind(next_buffer,current)]
  command=device.create_command_encoder()
  for step in range(steps):
   pass_gpu=command.begin_compute_pass()
   pass_gpu.set_pipeline(pipeline)
   pass_gpu.set_bind_group(0,binds[step%2])
   pass_gpu.dispatch_workgroups(math.ceil(n/128))
   pass_gpu.end()
  device.queue.submit([command.finish()])
  complete_buffer=next_buffer if steps%2 else current
  actual=np.frombuffer(device.queue.read_buffer(complete_buffer),
    dtype=np.float32).copy()
  elapsed_ms=(time.perf_counter()-begin)*1000
  max_error=float(np.max(np.abs(actual-expected)))
  nonfinite=int(np.count_nonzero(~np.isfinite(actual)))
  mean_error=float(np.mean(np.abs(actual-expected)))
  peak_speed=float(np.max(np.abs(actual.reshape(n,4)[:,2:])))
  passed=nonfinite==0 and max_error<=reference['tolerance'] and peak_speed<100
  result={'name':item['name'],'environments':n,'steps':steps,
   'actualJointStates':int(len(actual)),'ok':passed,
   'maxError':max_error,'meanError':mean_error,'nonfinite':nonfinite,
   'peakAngularSpeed':peak_speed,
   'synchronizedSetupSubmitAndReadbackMs':round(elapsed_ms,3)}
  print('ARTICULATED PHYSICAL GPU',json.dumps(result),flush=True)
  report['results'].append(result)
 report['ok']=len(report['results'])==3 and all(r['ok'] for r in report['results'])
 report['tolerance']=reference['tolerance']
 report['limitations']='Pinned hip, penalty foot force, feedback motors; no 3D free-base gait or RL training'
except Exception as error:
 report['error']=repr(error)
 report['traceback']=traceback.format_exc()[-5500:]
finally:
 report['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (out/'machines-native-hardware.json').write_text(json.dumps(report,indent=2))
 print('REAL_GPU_ARTICULATED_LEG_VERIFIED',report['ok'],flush=True)
 if not report['ok']:sys.exit(1)

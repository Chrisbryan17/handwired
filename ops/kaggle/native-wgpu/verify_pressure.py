"""Hardware-only native Vulkan WGSL differential for the actual Handwired
compiler's pressure computation. Distinct from browser WebGPU certification.
"""
import json,sys,os,time,traceback
from pathlib import Path
import numpy as np
import wgpu

base=Path(sys.argv[1] if len(sys.argv)>1 else '/kaggle/working')
result={'ok':False,'backend':'wgpu-native/Vulkan','browserWebGpuVerified':False,
 'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
try:
 os.environ['WGPU_BACKEND_TYPE']='Vulkan'
 # Insist on an actual T4 reported by wgpu-native itself. NVIDIA-SMI
 # alone does not establish which adapter executed this shader.
 adapters=wgpu.gpu.enumerate_adapters_sync()
 detected=[str(a.info) for a in adapters]
 result['visibleAdapters']=detected
 chosen=next((a for a in adapters
    if 'tesla t4' in str(a.info).lower()
    and not any(w in str(a.info).lower()
        for w in ('swiftshader','llvmpipe','lavapipe','software'))),None)
 if chosen is None:
  raise RuntimeError('No actual NVIDIA Tesla T4 Vulkan adapter in wgpu-native: '+str(detected))
 result['executingAdapter']=str(chosen.info)
 device=chosen.request_device_sync()
 reference=json.loads((base/'handwired-pressure-reference.json').read_text())
 shader_code=(base/'handwired-pressure.wgsl').read_text()
 rows=np.array(reference['rows'],dtype=np.float32)
 expected=np.array(reference['expected'],dtype=np.float32)
 assert rows.shape==(256,4) and expected.shape==(256,)
 U=wgpu.BufferUsage
 source=device.create_buffer_with_data(data=rows.tobytes(),
     usage=U.STORAGE|U.COPY_DST,label='Actual Handwired WGSL input')
 dest=device.create_buffer(size=expected.nbytes,
     usage=U.STORAGE|U.COPY_SRC,label='Actual native Vulkan pressure results')
 module=device.create_shader_module(code=shader_code,label='Handwired generated WGSL pressure')
 pipeline=device.create_compute_pipeline(
     layout='auto',compute={'module':module,'entry_point':'main'})
 bind=device.create_bind_group(
     layout=pipeline.get_bind_group_layout(0),entries=[
      {'binding':0,'resource':{'buffer':source}},
      {'binding':1,'resource':{'buffer':dest}},
     ])
 t0=time.perf_counter()
 encoder=device.create_command_encoder()
 compute=encoder.begin_compute_pass()
 compute.set_pipeline(pipeline)
 compute.set_bind_group(0,bind)
 compute.dispatch_workgroups(4)
 compute.end()
 device.queue.submit([encoder.finish()])
 completed_ms=(time.perf_counter()-t0)*1000
 read_start=time.perf_counter()
 output=np.frombuffer(device.queue.read_buffer(dest),dtype=np.float32).copy()
 read_ms=(time.perf_counter()-read_start)*1000
 invalid=int(np.count_nonzero(~np.isfinite(output)))
 max_error=float(np.max(np.abs(output-expected))) if len(output)==len(expected) else float('inf')
 mismatches=int(np.count_nonzero(np.abs(output-expected)>reference['tolerance'])) if len(output)==len(expected) else -1
 result.update({'ok':invalid==0 and mismatches==0 and len(output)==256,
  'numSamples':len(output),'maxError':max_error,'mismatches':mismatches,
  'invalid':invalid,'hostSubmitMs':completed_ms,'readbackMs':read_ms,
  'tolerance':reference['tolerance'],
  'shader':'Actual Handwired-compiled WGSL numerical circuit',
  'note':'True Vulkan GPU compute if executingAdapter is Tesla T4; not full SPH or browser WebGPU'})
except Exception as error:
 result['error']=repr(error)
 result['traceback']=traceback.format_exc()[-5000:]
finally:
 result['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (base/'handwired-native-wgsl-report.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 if not result['ok']:
  sys.exit(1)

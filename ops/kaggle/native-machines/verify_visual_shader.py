"""Hardware Vulkan compilation/link test for Neural Machines real 3D GPU
scene shaders. This does NOT render an image or replace browser visual QA.
"""
import wgpu,json,hashlib,os,sys,traceback,time
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else '/kaggle/working')
report={'ok':False,'renderExecuted':False,'browserWebGpuVerified':False,
 'backend':'wgpu-native/Vulkan','pipelines':[]}
try:
 os.environ['WGPU_BACKEND_TYPE']='Vulkan'
 adapters=wgpu.gpu.enumerate_adapters_sync()
 adapter=next((a for a in adapters if
 'tesla t4' in str(a.info).lower() and "backend_type='Vulkan'" in str(a.info)),None)
 if adapter is None:raise RuntimeError('No NVIDIA physical Vulkan adapter')
 report['adapter']=str(adapter.info)
 dev=adapter.request_device_sync()
 src=(root/'machines-visual.wgsl').read_text()
 report['sha256']=hashlib.sha256(src.encode()).hexdigest()
 shader=dev.create_shader_module(code=src,label='Actual Handwired physics-driven 3D robots')
 visibility=wgpu.ShaderStage
 layout=dev.create_bind_group_layout(entries=[
  {'binding':0,'visibility':visibility.VERTEX,
   'buffer':{'type':'read-only-storage'}},
  {'binding':1,'visibility':visibility.VERTEX|visibility.FRAGMENT,
   'buffer':{'type':'uniform'}},
 ])
 pipeline_layout=dev.create_pipeline_layout(bind_group_layouts=[layout])
 for entry in ('ground','rod','joint'):
  dev.create_render_pipeline(
   layout=pipeline_layout,
   vertex={'module':shader,'entry_point':entry,'buffers':[]},
   primitive={'topology':'triangle-list','cull_mode':'none'},
   depth_stencil={'format':'depth24plus','depth_write_enabled':True,
    'depth_compare':'less'},
   multisample={'count':1},
   fragment={'module':shader,'entry_point':'shadeMachine',
    'targets':[{'format':'rgba8unorm'}]})
  report['pipelines'].append(entry)
 report['ok']=len(report['pipelines'])==3
except Exception as error:
 report['error']=repr(error)
 report['traceback']=traceback.format_exc()[-5500:]
finally:
 (root/'machines-render-compile-report.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 if not report['ok']:sys.exit(1)

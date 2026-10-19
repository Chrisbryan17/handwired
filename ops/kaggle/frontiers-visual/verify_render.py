"""Hardware-compile and link every interactive Handwired frontier renderer.
Uses genuine Tesla T4 Vulkan adapter, not SwiftShader.
This checks WGSL and render pipeline compatibility, NOT browser visual QA.
"""
import json,os,sys,traceback,hashlib,time
from pathlib import Path
import wgpu
root=Path(sys.argv[1] if len(sys.argv)>1 else '/kaggle/working')
report={'ok':False,'browserWebGpuVerified':False,'offscreenPixelsVerified':False,
 'backend':'wgpu-native/Vulkan','pipelines':[],'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
try:
 os.environ['WGPU_BACKEND_TYPE']='Vulkan'
 adapters=wgpu.gpu.enumerate_adapters_sync()
 adapter=next((a for a in adapters if 'tesla t4' in str(a.info).lower()
    and "backend_type='Vulkan'" in str(a.info) and
    not any(x in str(a.info).lower() for x in ('swiftshader','llvmpipe','software'))),None)
 if adapter is None:raise RuntimeError('No real Tesla T4 Vulkan adapter')
 report['adapter']=str(adapter.info)
 device=adapter.request_device_sync()
 examples=[
  ('cataclysm-water',['vs'],'fs',[(0,'read-only-storage'),(1,'read-only-storage'),(2,'uniform')],True),
  ('destruction-bridge',['vs'],'fs',[(0,'read-only-storage'),(1,'read-only-storage'),(2,'read-only-storage'),(3,'uniform')],False),
  ('machines-robots',['ground','rod','joint'],'shadeMachine',[(0,'read-only-storage'),(1,'uniform')],False),
 ]
 for name,entries,fragment,bindings,frag_depth in examples:
  src=(root/(name+'.wgsl')).read_text()
  sha=hashlib.sha256(src.encode()).hexdigest()
  shader=device.create_shader_module(code=src,label=name+' source renderer')
  visibility=wgpu.ShaderStage
  group=device.create_bind_group_layout(entries=[
   {'binding':idx,'visibility':visibility.VERTEX|visibility.FRAGMENT,
    'buffer':{'type':kind}} for idx,kind in bindings])
  layout=device.create_pipeline_layout(bind_group_layouts=[group])
  for entry in entries:
   pipe=device.create_render_pipeline(layout=layout,
    vertex={'module':shader,'entry_point':entry,'buffers':[]},
    primitive={'topology':'triangle-list','cull_mode':'none'},
    depth_stencil={'format':'depth24plus','depth_write_enabled':True,'depth_compare':'less'},
    multisample={'count':1},
    fragment={'module':shader,'entry_point':fragment,
     'targets':[{'format':'rgba8unorm'}]})
   report['pipelines'].append({'workstream':name,'vertex':entry,'fragment':fragment,'sha256':sha,
    'hardwareLinked':True})
 report['ok']=len(report['pipelines'])==5
 report['limits']='Validates hardware shader module and pipeline linking; not pixels, user controls or browser execution'
except Exception as exc:
 report['error']=repr(exc);report['traceback']=traceback.format_exc()[-6000:]
finally:
 report['finished']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
 (root/'frontiers-native-visual-pipelines.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 if not report['ok']:sys.exit(1)

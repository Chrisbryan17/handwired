#!/usr/bin/env python3
"""One offloaded batch for Kaggle. No secrets are transmitted from the host."""
import json, os, pathlib, platform, shutil, subprocess, sys, time, urllib.request, zipfile
from datetime import datetime, timezone
OUT=pathlib.Path('/kaggle/working' if pathlib.Path('/kaggle/working').exists() else './kaggle-results')
OUT.mkdir(parents=True,exist_ok=True)
ROOT=OUT/'handwired'
BRANCH='feat/webgpu-kaggle-completion-20261016'
STATUS={'started':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'branch':BRANCH,'stages':[]}

def execute(name,argv,timeout=3600,optional=False,cwd=None):
    print('STAGE:',name,flush=True)
    start=time.monotonic()
    result={'name':name,'optional':optional,'args':argv}
    try:
        with (OUT/(name+'.log')).open('w') as log:
            child=subprocess.Popen(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,text=True)
            try:code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                code=124
        result['exit_code']=code
        result['tail']=(OUT/(name+'.log')).read_text(errors='replace')[-1500:]
    except Exception as exc:
        result['exit_code']=127
        result['error']=str(exc)
    result['seconds']=round(time.monotonic()-start,2)
    STATUS['stages'].append(result)
    (OUT/'summary.json').write_text(json.dumps(STATUS,indent=2))
    print(name,result['exit_code'],result.get('error',''),result.get('tail',''),flush=True)
    return result['exit_code']==0

def install_bun():
    dest=OUT/'bun'
    dest.mkdir(exist_ok=True)
    zip_path=dest/'bun.zip'
    urllib.request.urlretrieve(
        'https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-linux-x64.zip',zip_path)
    with zipfile.ZipFile(zip_path) as z:z.extractall(dest)
    binary=dest/'bun-linux-x64'/'bun'
    binary.chmod(0o755)
    return str(binary)

def main():
    if shutil.which('nvidia-smi'):execute('accelerator',['nvidia-smi','-L'],30,True)
    if not execute('checkout',['git','clone','--depth','1','--branch',BRANCH,
        'https://github.com/Chrisbryan17/handwired.git',str(ROOT)],180):return 2
    execute('revision',['git','rev-parse','HEAD'],15,True,ROOT)
    bun=install_bun()
    if not execute('dependencies',[bun,'install','--frozen-lockfile'],600,cwd=ROOT):return 3
    checks=[
        execute('targeted',[bun,'test','--timeout','30000','./tests/runtime/webgpu.test.ts',
          './tests/runtime/webgpu-completion.test.ts'],600,cwd=ROOT),
        execute('types-lint',[bun,'scripts/check.ts'],1800,cwd=ROOT),
        execute('full-suite',[bun,'test','--timeout','30000','./tests','./template','./lab'],
          4200,cwd=ROOT),
    ]
    chrome=next((x for x in ['chromium','chromium-browser','google-chrome'] if shutil.which(x)),None)
    if chrome:
        execute('browser-webgpu',[bun,'scripts/verify-webgpu-browser.ts',
          str(OUT/'browser-webgpu.json')],1200,True,ROOT)
    else:
        STATUS['stages'].append({'name':'browser-webgpu','not_run':
          'No Chromium binary; Kaggle CUDA GPU does not imply WebGPU browser access'})
    STATUS['completed']=datetime.now(timezone.utc).isoformat()
    STATUS['mandatory_passed']=all(checks)
    (OUT/'summary.json').write_text(json.dumps(STATUS,indent=2))
    return 0 if all(checks) else 4

if __name__=='__main__':sys.exit(main())

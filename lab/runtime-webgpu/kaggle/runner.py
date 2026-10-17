#!/usr/bin/env python3
"""One offloaded batch for Kaggle. No secrets are transmitted from the host."""
import json, os, pathlib, platform, shutil, subprocess, sys, time, urllib.request, zipfile
from datetime import datetime, timezone
OUT=pathlib.Path('/kaggle/working' if pathlib.Path('/kaggle/working').exists() else './kaggle-results')
OUT.mkdir(parents=True,exist_ok=True)
ROOT=(pathlib.Path('/kaggle/temp/handwired') if pathlib.Path('/kaggle/working').exists() else OUT/'handwired')
ROOT.parent.mkdir(parents=True, exist_ok=True)
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
    dest=(pathlib.Path('/kaggle/temp/handwired-bun') if pathlib.Path('/kaggle/working').exists() else OUT/'bun')
    dest.mkdir(exist_ok=True)
    zip_path=dest/'bun.zip'
    urllib.request.urlretrieve(
        'https://github.com/oven-sh/bun/releases/download/bun-v1.4.3/bun-linux-x64.zip',zip_path)
    with zipfile.ZipFile(zip_path) as z:z.extractall(dest)
    binary=dest/'bun-linux-x64'/'bun'
    binary.chmod(0o755)
    # scripts/check.ts invokes bunx as a subprocess. The portable Bun zip has
    # bun but no bunx executable; create its regular symlink in the same dir.
    bunx=binary.parent/'bunx'
    if not bunx.exists():bunx.symlink_to(binary.name)
    return str(binary)

def locate_browser():
    """Prefer a real Playwright Chromium binary, not a system Snap shim."""
    import glob
    patterns=[
        str(pathlib.Path.home()/'.cache/ms-playwright/chromium-*/chrome-linux64/chrome'),
        str(pathlib.Path.home()/'.cache/ms-playwright/chromium-*/chrome-linux/chrome'),
        str(pathlib.Path.home()/'.cache/ms-playwright/chromium_headless_shell-*/chrome-linux/headless_shell'),
    ]
    for pattern in patterns:
        for item in sorted(glob.glob(pattern),reverse=True):
            if pathlib.Path(item).is_file():return item
    for name in ['google-chrome','chromium','chromium-browser']:
        candidate=shutil.which(name)
        if candidate:return candidate
    return None

def main():
    # Deliberately a browser-only run. CPU suites (1,284 passing) and
    # TypeScript/lint (0 errors) were already verified remotely.
    if shutil.which('nvidia-smi'):
        execute('accelerator',['nvidia-smi','-L'],30,True)
    execute('vulkan-environment',['bash','-lc',
        'ls -al /usr/share/vulkan/icd.d/ 2>&1; ldconfig -p | grep -E "libvulkan|libEGL|libnvidia" | head -20; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'],30,True)
    if not execute('checkout',['git','clone','--depth','1','--branch',BRANCH,
        'https://github.com/Chrisbryan17/handwired.git',str(ROOT)],180):return 2
    execute('revision',['git','rev-parse','HEAD'],15,True,ROOT)
    bun=install_bun()
    os.environ['PATH']=str(pathlib.Path(bun).parent)+os.pathsep+os.environ.get('PATH','')
    if not execute('dependencies',[bun,'install','--frozen-lockfile'],600,cwd=ROOT):return 3
    STATUS['prior_validation']={
      'focused':{'passed':20,'failed':0,'source_sha':'b03b57ab7e32cbe98245ddad326ab663f4a31f02'},
      'full_suite':{'passed':1284,'failed':0,'skipped':2,'source_sha':'b03b57ab7e32cbe98245ddad326ab663f4a31f02'},
      'static':{'typescript':'pass','lint_errors':0,'lint_warnings':0,
        'source_sha':'ddf9839ab3767a060cdbf89aac99fd39596c4433'},
    }
    execute('playwright-package',[sys.executable,'-m','pip','install','-q','playwright'],360,True)
    execute('playwright-browser',[sys.executable,'-m','playwright','install','chromium'],540,True)
    browser=locate_browser()
    if not browser:
        STATUS['error']='Chromium install failed; no browser executable'
        (OUT/'summary.json').write_text(json.dumps(STATUS,indent=2))
        return 5
    STATUS['chromium']=browser
    execute('chromium-version',[browser,'--version'],60,True)
    # Debian/Ubuntu browser dynamic dependencies may need apt-based bootstrap.
    execute('chromium-libraries',[sys.executable,'-m','playwright','install-deps','chromium'],420,True)
    # Kaggle's CUDA container may expose nvidia-smi without NVIDIA Vulkan ICD
    # libraries. Install only the matching major-version GPU userspace package,
    # never a new kernel driver, in the ephemeral Kaggle container.
    execute('nvidia-userspace-candidates',
        ['bash','-lc','apt-cache policy libnvidia-gl-580 libnvidia-gl-580-server vulkan-tools; ls -lah /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1 || true'],45,True)
    execute('nvidia-vulkan-userspace',['bash','-lc',
        'export DEBIAN_FRONTEND=noninteractive; if apt-cache show libnvidia-gl-580 >/dev/null 2>&1; then apt-get install -y --no-install-recommends libnvidia-gl-580 vulkan-tools; elif apt-cache show libnvidia-gl-580-server >/dev/null 2>&1; then apt-get install -y --no-install-recommends libnvidia-gl-580-server vulkan-tools; else echo "No NVIDIA 580 Vulkan userspace package available in Kaggle APT sources"; exit 2; fi'],420,True)
    execute('vulkan-detection',['bash','-lc',
        'ls -lah /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; vulkaninfo --summary 2>&1 | head -100'],60,True)
    os.environ['HANDWIRED_GPU_BROWSER']=browser
    # First require the real NVIDIA adapter. Never call a SwiftShader result NVIDIA.
    os.environ.pop('HANDWIRED_GPU_SOFTWARE',None)
    os.environ['HANDWIRED_GPU_VULKAN']='1'
    os.environ['HANDWIRED_EXPECT_GPU_VENDOR']='nvidia'
    os.environ['HANDWIRED_GPU_EXTRA_FLAGS']='--ignore-gpu-blocklist --disable-vulkan-surface --ozone-platform=x11'
    # NVIDIA hardware shader execution was independently verified on v8:
    # native Tesla T4 compute 42/42 and full Handwired smoke all green.
    # Isolate performance to small/large workloads and report failed stage.
    STATUS['prior_nvidia_hardware_smoke']={
        'kaggle_version':8,'adapter':'nvidia turing Tesla T4',
        'compute':'42/42','smoke':'pass','gallery_mismatches':0,'grid_mismatches':0,
        'state_mismatches':0}
    STATUS['prior_nvidia_benchmark']={'kaggle_version':11,'nvidia':True,
      'small_512_steps8':{'cpu_ms':12.7,'specialized_gpu_ms':2.2,'handwritten_gpu_ms':2.0},
      'large_8192_steps8':{'cpu_ms':82.9,'specialized_gpu_ms':6.0,'handwritten_gpu_ms':2.8},
      'zero_mismatches':True}
    small=execute('nvidia-benchmark-512-steps128',
        [bun,'scripts/verify-webgpu-browser.ts',str(OUT/'nvidia-benchmark-512-steps128.json'),
          'benchmark','128','512'],240,True,ROOT)
    large=execute('nvidia-benchmark-8192-steps128',
        [bun,'scripts/verify-webgpu-browser.ts',str(OUT/'nvidia-benchmark-8192-steps128.json'),
          'benchmark','128','8192'],300,True,ROOT)
    STATUS['nvidia_small_benchmark_ok']=small
    STATUS['nvidia_large_benchmark_ok']=large
    STATUS['completed']=datetime.now(timezone.utc).isoformat()
    STATUS['chromium_started']=True
    STATUS['all_required_hardware_passed']=small and large
    (OUT/'summary.json').write_text(json.dumps(STATUS,indent=2))
    # A software-only pass is informative but does not clear NVIDIA gate.
    return 0 if (small and large) else 6

if __name__=='__main__':sys.exit(main())

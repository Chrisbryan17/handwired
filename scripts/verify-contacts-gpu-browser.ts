
/** Kill the entire ephemeral Chromium process tree on Windows.
 * Killing only the spawned root leaves GPU, renderer and crashpad child processes
 * alive across test runs, eventually exhausting the host's OS thread capacity.
 */
function terminateHeadlessBrowser(child: ReturnType<typeof Bun.spawn> | undefined,
  browserProfile: string): void {
  if (child === undefined) return
  if (process.platform !== 'win32') {child.kill(); return}
  // The root process can exit while its GPU, renderer and crashpad children
  // remain orphaned. Process-tree termination alone is insufficient.
  try {
    Bun.spawnSync(['taskkill.exe', '/T', '/F', '/PID', String(child.pid)],
      {stdout:'ignore',stderr:'ignore'})
  } catch {}
  // Restrict cleanup to this verifier's unique temporary browser profile;
  // this cannot match the user's normal browsing profile.
  const escapedProfile = browserProfile.replaceAll("'", "''")
  const script = "$profile='" + escapedProfile + "';" +
    'Get-CimInstance Win32_Process -Filter "Name=\'chrome.exe\' OR Name=\'msedge.exe\'" ' +
    '| Where-Object { $_.CommandLine -like ("*" + $profile + "*") } ' +
    '| ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }'
  try {
    Bun.spawnSync(['powershell.exe', '-NoProfile', '-NonInteractive',
      '-Command', script], {stdout:'ignore',stderr:'ignore'})
  } catch {}
  child.kill()
}

// Real Chromium WebGPU smoke, independent CPU oracle comparison and screenshot.
// Rejects software GPU identity and never substitutes compilation for execution.
import page from '../lab/frontiers/contact-gpu/probe.html'
import {mkdtempSync,mkdirSync,rmSync,writeFileSync} from 'node:fs'
import {dirname,join} from 'node:path'
import {tmpdir} from 'node:os'
const report=process.argv[2]??'lab/frontiers/contact-gpu/results/hardware-probe.json'
const server=Bun.serve({hostname:'127.0.0.1',port:39169,routes:{'/':page}})
const profile=mkdtempSync(join(tmpdir(),'handwired-contact-'))
const port=49589
const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms))
const executables=process.platform==='win32'
  ? ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe']
  : ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser']
if(process.env['HANDWIRED_GPU_BROWSER'] !== undefined && process.env['HANDWIRED_GPU_BROWSER'] !== '')executables.unshift(process.env['HANDWIRED_GPU_BROWSER'])
let browser:ReturnType<typeof Bun.spawn>|undefined
async function run(){
  let pageTarget:{webSocketDebuggerUrl:string}|undefined
  let executable=''
  for(const candidate of executables){
    try{
      const child=Bun.spawn([candidate,'--headless=new','--no-first-run','--no-sandbox',
        '--disable-gpu-sandbox','--disable-dev-shm-usage','--enable-unsafe-webgpu',
        '--enable-webgpu-developer-features',
     ...(process.env['HANDWIRED_BROWSER_FLAGS']??'').split(' ').filter(Boolean),
     '--window-size=1440,900',
        '--remote-allow-origins=*','--disable-extensions',
        '--remote-debugging-port='+port,'--user-data-dir='+profile,'about:blank'],
        {stdout:'ignore',stderr:'ignore'})
      for(let i=0;i<60;i++){
        await delay(250)
        try{
          const response=await fetch('http://127.0.0.1:'+port+'/json/list')
          if(!response.ok)continue
          const targets=await response.json() as {type:string,webSocketDebuggerUrl:string}[]
          pageTarget=targets.find(t=>t.type==='page')
          if(pageTarget){browser=child;executable=candidate;break}
        }catch{}
      }
      if(pageTarget)break
      terminateHeadlessBrowser(child, profile)
    }catch{}
  }
  if(!pageTarget)throw new Error('No CDP-enabled Chromium / Edge browser')
  const socket=new WebSocket(pageTarget.webSocketDebuggerUrl)
  const pending=new Map<number,(value:{result?:Record<string,unknown>,error?:unknown})=>void>()
  let id=0
  socket.onmessage=e=>{
    const message=JSON.parse(String(e.data)) as {id?:number,result?:Record<string,unknown>,error?:unknown}
    if(message.id!==undefined){pending.get(message.id)?.(message);pending.delete(message.id)}
  }
  await new Promise<void>((resolve,reject)=>{
    socket.onopen=()=>resolve()
    socket.onerror=()=>reject(new Error('CDP websocket error'))
  })
  const cmd=async(method:string,params:Record<string,unknown>={})=>{
    const key=++id
    const promise=new Promise<{result?:Record<string,unknown>,error?:unknown}>(resolve=>pending.set(key,resolve))
    socket.send(JSON.stringify({id:key,method,params}))
    const answer=await promise
    if(answer.error!==undefined)throw new Error('CDP '+JSON.stringify(answer.error))
    return answer.result
  }
  const evaluate=async(expression:string)=>{
    const res=await cmd('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true})
    const reply=res?.['result'] as {value?:unknown,description?:string}|undefined
    if(res?.['exceptionDetails'] !== undefined)throw new Error('Browser script: '+JSON.stringify(res['exceptionDetails']))
    return reply?.value
  }
  try{
    await cmd('Page.enable')
    await cmd('Runtime.enable')
    const mobile=process.argv.includes('--mobile')
    await cmd('Emulation.setDeviceMetricsOverride',{width:mobile?390:1440,
      height:mobile?844:900,deviceScaleFactor:mobile?2:1,mobile})
    await cmd('Page.navigate',{url:'http://127.0.0.1:39169/'})
    let reportData: Record<string,unknown> | null = null
    for(let poll=0;poll<300;poll++){
      await delay(250)
      const raw=String(await evaluate(`document.querySelector('#result')?.textContent || ''`))
      if(raw.length<2)continue
      const parsed=JSON.parse(raw) as Record<string,unknown>
      reportData=parsed
      break
    }
    if(reportData===null)throw new Error('Hardware benchmark timed out')
    const adapter=await evaluate(`(async()=>{const a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});return a?{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,isFallbackAdapter:a.isFallbackAdapter??null}:null})()`) as Record<string,unknown>|null
    const identity=adapter?[adapter['vendor'],adapter['architecture'],adapter['device'],adapter['description']].join(' '):''
    const hardware=Boolean(adapter && adapter['isFallbackAdapter']!==true &&
      typeof adapter['vendor']==='string' && adapter['vendor']!=='' &&
      !/swiftshader|llvmpipe|lavapipe|software|softpipe|subzero/i.test(identity))

    const cases=Array.isArray(reportData['results'])?reportData['results'] as Array<Record<string,unknown>>:[]
    const validPath=reportData['ok']===true && cases.length===2 &&
      cases.every(c=>c['passes']===true && Number(c['maxError'])<=.002)
    const summary={ok:validPath&&hardware,date:new Date().toISOString(),
      browser:executable,actualHardwareAdapter:adapter,hardware,
      test:'GPU sphere contacts, spatial broadphase and independent CPU Jacobi physics parity',
      ...reportData}
    if(process.argv.includes('--screenshot')){
      const shot=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})
      const base64=shot?.['data']
      if(typeof base64==='string')
        writeFileSync(report.replace(/\.json$/,'.png'),Buffer.from(base64,'base64'))
    }
    return summary
  }finally{socket.close()}
}
try{
  mkdirSync(dirname(report),{recursive:true})
  const summary=await run()
  writeFileSync(report,JSON.stringify(summary,null,2)+'\n')
  console.log(JSON.stringify(summary,null,2))
  if(!summary.ok)process.exitCode=1
}catch(error){
  const summary={ok:false,date:new Date().toISOString(),error:String(error)}
  mkdirSync(dirname(report),{recursive:true})
  writeFileSync(report,JSON.stringify(summary,null,2)+'\n')
  console.error(summary)
  process.exitCode=1
}finally{
  terminateHeadlessBrowser(browser, profile)
  try{rmSync(profile,{recursive:true,force:true})}catch{}
  void server.stop(true)
}
process.exit(process.exitCode??0)


/** Kill the entire ephemeral Chromium process tree on Windows.
 * Killing only the spawned root leaves GPU, renderer and crashpad child processes
 * alive across test runs, eventually exhausting the host's OS thread capacity.
 */
function terminateHeadlessBrowser(child: ReturnType<typeof Bun.spawn> | undefined): void {
  if (child === undefined) return
  if (process.platform !== 'win32') {child.kill(); return}
  try {
    const taskkill = Bun.spawnSync(
      ['taskkill.exe', '/T', '/F', '/PID', String(child.pid)],
      {stdout:'ignore',stderr:'ignore'},
    )
    if (taskkill.exitCode !== 0) child.kill()
  } catch {
    child.kill()
  }
}

// Real Chromium WebGPU smoke, independent CPU oracle comparison and screenshot.
// Rejects software GPU identity and never substitutes compilation for execution.
import page from '../lab/frontiers/genesis/demo.html'
import {mkdtempSync,mkdirSync,rmSync,writeFileSync} from 'node:fs'
import {dirname,join} from 'node:path'
import {tmpdir} from 'node:os'
const report=process.argv[2]??'lab/frontiers/genesis/results/hardware-smoke.json'
const server=Bun.serve({hostname:'127.0.0.1',port:39140,routes:{'/':page}})
const profile=mkdtempSync(join(tmpdir(),'handwired-genesis-'))
const port=49560
const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms))
const executables=process.platform==='win32'
  ? ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe']
  : ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser']
if(process.env['HANDWIRED_GPU_BROWSER'])executables.unshift(process.env['HANDWIRED_GPU_BROWSER'])
let browser:ReturnType<typeof Bun.spawn>|undefined
async function run(){
  let pageTarget:{webSocketDebuggerUrl:string}|undefined
  let executable=''
  for(const candidate of executables){
    try{
      const child=Bun.spawn([candidate,'--headless=new','--no-first-run','--no-sandbox',
        '--disable-gpu-sandbox','--disable-dev-shm-usage','--enable-unsafe-webgpu',
        '--enable-webgpu-developer-features','--window-size=1440,900',
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
      terminateHeadlessBrowser(child)
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
    if(res?.['exceptionDetails'])throw new Error('Browser script: '+JSON.stringify(res['exceptionDetails']))
    return reply?.value
  }
  try{
    await cmd('Page.enable')
    await cmd('Runtime.enable')
    await cmd('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false})
    await cmd('Page.navigate',{url:'http://127.0.0.1:39140/'})
    let state:{chip:string,step:number,error:string}|undefined
    for(let i=0;i<120;i++){
      await delay(250)
      const value=await evaluate(`({chip:document.querySelector('#gpu')?.textContent||'',step:Number(document.querySelector('#steps')?.textContent?.replace(/,/g,'')||0),error:document.querySelector('#error')?.hidden?'':document.querySelector('#error')?.textContent||''})`)
      state=value as {chip:string,step:number,error:string}
      if(state.error)throw new Error('Browser demo: '+state.error)
      if(state.step>=12)break
    }
    if(!state||state.step<12)throw new Error('WebGPU compute did not advance 12 displayed steps: '+JSON.stringify(state))
    await evaluate(`document.querySelector('#pause')?.click()`)
    await evaluate(`document.querySelector('#verify')?.click()`)
    let verify=''
    for(let i=0;i<120;i++){
      await delay(250)
      verify=String(await evaluate(`document.querySelector('#verification')?.textContent||''`))
      if(verify.startsWith('PASS')||verify.startsWith('FAIL')||verify.startsWith('Verification failed'))break
    }
    const adapter=await evaluate(`(async()=>{let a=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});return a?{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,isFallbackAdapter:a.isFallbackAdapter??null}:null})()`) as Record<string,unknown>|null
    const label=adapter?[adapter['vendor'],adapter['architecture'],adapter['device'],adapter['description']].join(' '):''
    const software=!adapter||adapter['isFallbackAdapter']===true||
      /swiftshader|llvmpipe|lavapipe|softpipe|software|subzero/i.test(label)
    const gpuHardwareIdentified=Boolean(adapter&&typeof adapter['vendor']==='string'&&adapter['vendor']!==''&&!software)
    const success=verify.startsWith('PASS')&&state.step>=12&&gpuHardwareIdentified
    const summary={ok:success,date:new Date().toISOString(),browser:executable,
      test:'Genesis 12+ WebGPU chemistry+Handwired steps and one independent CPU chemical parity',
      adapter,software,gpuHardwareIdentified,stepsBeforeVerification:state.step,
      verification:verify,frameCadenceNotKernelTime:true}
    if(process.argv.includes('--screenshot')){
      const response=await cmd('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})
      const base64=response?.['data']
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
  terminateHeadlessBrowser(browser)
  try{rmSync(profile,{recursive:true,force:true})}catch{}
  void server.stop(true)
}
process.exit(process.exitCode??0)

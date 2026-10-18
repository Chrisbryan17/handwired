import page from '../lab/frontiers/light/demo.html'
import {mkdtempSync,mkdirSync,rmSync,writeFileSync} from 'node:fs'
import {dirname,join} from 'node:path'
import {tmpdir} from 'node:os'
/** Real browser render test, not merely shader compilation; every frame
 * advances WebGPU radiance accumulation and display on a physical adapter.
 * The independent CPU oracle is in light/model.test.ts.
 */
const report=process.argv[2]??'lab/frontiers/light/results/demo-hardware.json'
const profile=mkdtempSync(join(tmpdir(),'handwired-light-demo-'))
const devtoolsPort=49568
const server=Bun.serve({hostname:'127.0.0.1',port:39148,routes:{'/':page}})
const delay=(ms:number)=>new Promise(resolve=>setTimeout(resolve,ms))
const choices=process.platform==='win32'?
 ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe','C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe']:
 ['/usr/bin/google-chrome','/usr/bin/chromium','/usr/bin/chromium-browser']
if(process.env['HANDWIRED_GPU_BROWSER'] !== undefined && process.env['HANDWIRED_GPU_BROWSER'] !== '')choices.unshift(process.env['HANDWIRED_GPU_BROWSER'])
let browser:ReturnType<typeof Bun.spawn>|undefined
function cleanup(){
 if(!browser)return
 if(process.platform==='win32'){
  try{Bun.spawnSync(['taskkill.exe','/T','/F','/PID',String(browser.pid)],
    {stdout:'ignore',stderr:'ignore'})}catch{}
  const escaped=profile.replaceAll("'","''")
  const ps="$p='"+escaped+"';Get-CimInstance Win32_Process -Filter \"Name='chrome.exe' OR Name='msedge.exe'\" | Where-Object {$_.CommandLine -like ('*'+$p+'*')} | ForEach-Object {Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue}"
  try{Bun.spawnSync(['powershell.exe','-NoProfile','-NonInteractive','-Command',ps],
    {stdout:'ignore',stderr:'ignore'})}catch{}
 }
 browser.kill()
}
async function run(){
 let target:{webSocketDebuggerUrl:string}|undefined,executable=''
 for(const candidate of choices){
  try{
   const child=Bun.spawn([candidate,'--headless=new','--no-first-run','--no-sandbox',
     '--disable-gpu-sandbox','--disable-dev-shm-usage','--enable-unsafe-webgpu',
     '--enable-webgpu-developer-features','--window-size=1440,900',
     '--remote-allow-origins=*','--disable-extensions',
     '--remote-debugging-port='+devtoolsPort,'--user-data-dir='+profile,'about:blank'],
     {stdout:'ignore',stderr:'ignore'})
   for(let i=0;i<55;i++){
    await delay(250)
    try{
     const res=await fetch('http://127.0.0.1:'+devtoolsPort+'/json/list')
     if(!res.ok)continue
     const list=await res.json() as {type:string,webSocketDebuggerUrl:string}[]
     target=list.find(item=>item.type==='page')
     if(target){browser=child;executable=candidate;break}
    }catch{}
   }
   if(target)break
   child.kill()
  }catch{}
 }
 if(!target)throw new Error('No Chromium DevTools browser')
 const ws=new WebSocket(target.webSocketDebuggerUrl)
 const pending=new Map<number,(response:{result?:Record<string,unknown>,error?:unknown})=>void>()
 let count=0
 ws.onmessage=e=>{
  const msg=JSON.parse(String(e.data)) as {id?:number,result?:Record<string,unknown>,error?:unknown}
  if(msg.id!==undefined){pending.get(msg.id)?.(msg);pending.delete(msg.id)}
 }
 await new Promise<void>((resolve,reject)=>{
  ws.onopen=()=>resolve()
  ws.onerror=()=>reject(new Error('Browser WebSocket connection failed'))
 })
 const call=async(method:string,params:Record<string,unknown>={})=>{
  const id=++count
  const p=new Promise<{result?:Record<string,unknown>,error?:unknown}>(resolve=>pending.set(id,resolve))
  ws.send(JSON.stringify({id,method,params}))
  const value=await p
  if(value.error!==undefined)throw new Error('CDP '+JSON.stringify(value.error))
  return value.result
 }
 const evaluate=async(expression:string)=>{
  const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true})
  if(r?.['exceptionDetails'] !== undefined)throw new Error(JSON.stringify(r['exceptionDetails']))
  return (r?.['result'] as {value?:unknown}|undefined)?.value
 }
 try{
  const mobile=process.argv.includes('--mobile')
  await call('Page.enable');await call('Runtime.enable')
  await call('Emulation.setDeviceMetricsOverride',{width:mobile?390:1440,
    height:mobile?844:900,deviceScaleFactor:mobile?2:1,mobile})
  await call('Page.navigate',{url:'http://127.0.0.1:39148/'})
  const targetSamples=Math.max(36,Math.min(1024,Number(process.argv.find(a=>a.startsWith('--samples='))?.split('=')[1]??'36')))
  if(!Number.isInteger(targetSamples))throw new Error('bad sample target')
  let snapshot:{status:string,samples:number,error:string,bytes:string,pixels:string}|undefined
  for(let i=0;i<800;i++){
   await delay(250)
   snapshot=await evaluate(`({status:document.querySelector('#gpu-status')?.textContent||'',
   samples:Number(document.querySelector('#spp')?.textContent?.replace(/,/g,'')||0),
   error:document.querySelector('#error')?.hidden?'':document.querySelector('#error')?.textContent||'',
   bytes:document.querySelector('#bytes')?.textContent||'',
   pixels:document.querySelector('#pixels')?.textContent||''})`) as typeof snapshot
   if(snapshot?.error !== undefined && snapshot.error.length > 0)throw new Error('Demo: '+snapshot.error)
   if((snapshot?.samples??0)>=targetSamples)break
  }
  if(!snapshot||snapshot.samples<targetSamples)throw new Error('Progressive radiance did not reach '+targetSamples+' samples')
  const adapter=await evaluate(`(async()=>{const a=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});return a?{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,isFallbackAdapter:a.isFallbackAdapter??null}:null})()`) as Record<string,unknown>|null
  const label=adapter?[adapter['vendor'],adapter['architecture'],adapter['device'],adapter['description']].join(' '):''
  const hardware=Boolean(adapter&&adapter['isFallbackAdapter']!==true&&
   typeof adapter['vendor']==='string'&&adapter['vendor']!==''&&
   !/swiftshader|llvmpipe|lavapipe|software|softpipe|subzero/i.test(label))
  const valid=snapshot.status.includes('GPU PATH TRACING ACTIVE')&&hardware
  if(process.argv.includes('--screenshot')){
   await evaluate(`document.querySelector('#pause')?.click()`)
   await delay(120)
   const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})
   const data=screenshot?.['data']
   if(typeof data!=='string')throw new Error('CDP did not return a screenshot')
   writeFileSync(report.replace(/\.json$/,'.png'),Buffer.from(data,'base64'))
  }
  return {ok:valid,date:new Date().toISOString(),browser:executable,
   viewport:mobile?'390x844-mobile':'1440x900-desktop',
   adapter,hardware,targetSamples,snapshot,meaning:'Real progressive GPU render; no CPU path-tracing parity implied',
   researchBoundaries:'Unaccelerated sphere/plane intersections, no BVH/MIS/denoiser'}
 }finally{ws.close()}
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
 console.error(summary);process.exitCode=1
}finally{
 cleanup()
 try{rmSync(profile,{recursive:true,force:true})}catch{}
 void server.stop(true)
}
process.exit(process.exitCode??0)

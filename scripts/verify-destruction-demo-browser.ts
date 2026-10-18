import page from '../lab/frontiers/destruction/demo.html'
import {mkdtempSync,mkdirSync,rmSync,writeFileSync} from 'node:fs'
import {dirname,join} from 'node:path'
import {tmpdir} from 'node:os'
/** Real browser render test, not merely shader compilation; every frame
 * advances WebGPU radiance accumulation and display on a physical adapter.
 * The independent CPU oracle is in light/model.test.ts.
 */
const report=process.argv[2]??'lab/frontiers/destruction/results/demo-hardware.json'
const profile=mkdtempSync(join(tmpdir(),'handwired-destruction-demo-'))
const devtoolsPort=49620
const server=Bun.serve({hostname:'127.0.0.1',port:39179,routes:{'/':page}})
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
  const deployedUrl='https://chrisbryan17.github.io/handwired/neural-light/'
  const currentUrl=process.argv.includes('--public')?deployedUrl:'http://127.0.0.1:39179/'
  await call('Page.navigate',{url:currentUrl})
  const targetSamples=Math.max(20,Math.min(240,Number(process.argv.find(a=>a.startsWith('--steps='))?.split('=')[1]??'90')))
  if(!Number.isInteger(targetSamples))throw new Error('bad physics step target')
  let snapshot:{status:string,samples:number,error:string,bytes:string,pixels:string}|undefined
  for(let i=0;i<800;i++){
   await delay(250)
   snapshot=await evaluate(`({status:document.querySelector('#gpu-status')?.textContent||'',
   samples:Number(document.querySelector('#step-count')?.textContent?.replace(/,/g,'')||0),
   error:document.querySelector('#error')?.hidden?'':document.querySelector('#error')?.textContent||'',
   bytes:document.querySelector('#memory')?.textContent||'',
   pixels:document.querySelector('#body-count')?.textContent||''})`) as typeof snapshot
   if(snapshot?.error !== undefined && snapshot.error.length > 0)throw new Error('Demo: '+snapshot.error)
   if((snapshot?.samples??0)>=targetSamples)break
  }
  if(!snapshot||snapshot.samples<targetSamples)throw new Error('Sphere contact physics did not reach '+targetSamples+' samples')
  const adapter=await evaluate(`(async()=>{const a=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});return a?{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,isFallbackAdapter:a.isFallbackAdapter??null}:null})()`) as Record<string,unknown>|null
  const label=adapter?[adapter['vendor'],adapter['architecture'],adapter['device'],adapter['description']].join(' '):''
  const hardware=Boolean(adapter&&adapter['isFallbackAdapter']!==true&&
   typeof adapter['vendor']==='string'&&adapter['vendor']!==''&&
   !/swiftshader|llvmpipe|lavapipe|software|softpipe|subzero/i.test(label))
  await evaluate(`document.querySelector('#pause')?.click();document.querySelector('#verify')?.click();true`)
  let verifyResult=''
  for(let i=0;i<100;i++){
   await delay(100)
   verifyResult=String(await evaluate(`document.querySelector('#verification')?.textContent||''`))
   if(verifyResult.startsWith('PASS')||verifyResult.startsWith('FAIL')||
     verifyResult.startsWith('Verification failed'))break
  }
  if(!verifyResult.startsWith('PASS'))throw new Error('Interactive CPU/GPU parity check: '+verifyResult)
  const valid=snapshot.status.includes('PHYSICS COMPUTE ACTIVE')&&hardware

  if(process.argv.includes('--screenshot')){
   await evaluate(`document.querySelector('#pause')?.click()`)
   await delay(120)
   const screenshot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})
   const data=screenshot?.['data']
   if(typeof data!=='string')throw new Error('CDP did not return a screenshot')
   writeFileSync(report.replace(/\.json$/,'.png'),Buffer.from(data,'base64'))
  }
  return {ok:valid,date:new Date().toISOString(),browser:executable,interactiveVerification:verifyResult,publicRoute:process.argv.includes('--public')?currentUrl:null,
   viewport:mobile?'390x844-mobile':'1440x900-desktop',
   adapter,hardware,targetSamples,snapshot,meaning:'Real 3D GPU sphere contact simulation; explicit independent CPU contact verification runs during this probe',
   researchBoundaries:'Sphere rigid contacts only, no fracture, joints, stress tensor, broad materials or CCD'}
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

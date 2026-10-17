// Headless, hardware-backed smoke and visual QA for the neural swarm experiment.
// Usage: bun scripts/verify-visual-swarm-browser.ts [report.json] [count] [--screenshot]
// Unlike CPU tests, this requires actual browser WebGPU and rejects a CPU fallback.
import page from '../lab/visual-swarm/demo.html'
import {mkdtempSync, mkdirSync, rmSync, writeFileSync} from 'node:fs'
import {dirname, join} from 'node:path'
import {tmpdir} from 'node:os'

const report = process.argv[2] ?? 'lab/visual-swarm/results/smoke.json'
const count = Math.max(1, Math.min(8192, Number(process.argv[3] ?? '128')))
const server = Bun.serve({hostname: '127.0.0.1', port: 39082, routes: {'/': page}})
const profile = mkdtempSync(join(tmpdir(), 'handwired-swarm-'))
const devtoolsPort = 49383
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))
const candidates = process.platform === 'win32'
  ? ['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe']
  : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser']
if (process.env['HANDWIRED_GPU_BROWSER'] !== undefined && process.env['HANDWIRED_GPU_BROWSER'] !== '') candidates.unshift(process.env['HANDWIRED_GPU_BROWSER'])
let browser: ReturnType<typeof Bun.spawn> | undefined

async function run() {
  let target: {webSocketDebuggerUrl: string} | undefined
  let executable = ''
  for (const candidate of candidates) {
    try {
      const child = Bun.spawn([candidate, '--headless=new', '--no-first-run',
        '--no-sandbox', '--disable-gpu-sandbox', '--disable-dev-shm-usage',
        '--enable-unsafe-webgpu', '--enable-webgpu-developer-features', '--window-size=1440,900',
        ...(process.env['HANDWIRED_GPU_VULKAN'] === '1'
          ? ['--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan,WebGPU',
            '--use-gl=angle'] : []),
        '--remote-allow-origins=*', '--disable-extensions',
        '--remote-debugging-port=' + devtoolsPort,
        '--user-data-dir=' + profile, 'about:blank'], {stdout:'ignore',stderr:'ignore'})
      for (let k = 0; k < 45; k++) {
        await delay(250)
        try {
          const response = await fetch('http://127.0.0.1:' + devtoolsPort + '/json/list')
          if (!response.ok) continue
          const list = await response.json() as {type: string,webSocketDebuggerUrl: string}[]
          target = list.find(item => item.type === 'page')
          if (target) {browser = child; executable = candidate; break}
        } catch {}
      }
      if (target) break
      child.kill()
    } catch {}
  }
  if (!target) throw new Error('Chrome/Edge DevTools could not start')
  const socket = new WebSocket(target.webSocketDebuggerUrl)
  const pending = new Map<number, (value: {result?: Record<string, unknown>, error?: unknown}) => void>()
  let nextId = 0
  socket.onmessage = event => {
    const message = JSON.parse(String(event.data)) as {
      id?: number, result?: Record<string,unknown>, error?:unknown
    }
    if (message.id !== undefined) {
      pending.get(message.id)?.(message)
      pending.delete(message.id)
    }
  }
  await new Promise<void>((resolve,reject)=>{
    socket.onopen=()=>resolve()
    socket.onerror=()=>reject(new Error('DevTools websocket error'))
  })
  const command = async (method:string,params:Record<string,unknown>={}) => {
    const id=++nextId
    const result=new Promise<{result?:Record<string,unknown>,error?:unknown}>(
      resolve=>pending.set(id,resolve))
    socket.send(JSON.stringify({id,method,params}))
    const answer=await result
    if(answer.error !== undefined && answer.error !== null)throw new Error(JSON.stringify(answer.error))
    return answer.result
  }
  try {
    await command('Page.enable')
    await command('Runtime.enable')
    await command('Page.navigate',{url:'http://127.0.0.1:39082/?smoke=1&count='+count})
    let smoke: Record<string,unknown> | null = null
    for(let k=0;k<240;k++) {
      await delay(250)
      const response=await command('Runtime.evaluate',{
        expression:'JSON.stringify({smoke:document.querySelector("#smoke")?.textContent, stage:document.querySelector("#status")?.textContent})',
        returnByValue:true,
      })
      const text=(response?.['result'] as {value?: string}|undefined)?.value
      if (text === undefined || text === '') continue
      const current=JSON.parse(text) as {smoke?:string,stage?:string}
      if (current.smoke !== undefined && current.smoke !== '') {smoke=JSON.parse(current.smoke) as Record<string,unknown>;break}
    }
    if (!smoke) throw new Error('Swarm smoke timed out before eight completed frames')
    const answer=await command('Runtime.evaluate',{
      expression:'(async()=>{const a=await navigator.gpu?.requestAdapter({powerPreference:"high-performance"});return a?{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description,isFallbackAdapter:a.isFallbackAdapter??null}:{error:"no adapter available after test"}})()',
      awaitPromise:true,returnByValue:true,
    })
    const adapter=(answer?.['result'] as {value?:Record<string,unknown>}|undefined)?.value ?? {}
    const label=[adapter['vendor'],adapter['architecture'],adapter['device'],adapter['description']]
      .join(' ').toLowerCase()
    const software=adapter['isFallbackAdapter']===true ||
      /swiftshader|llvmpipe|lavapipe|softpipe|software|subzero/.test(label)
    const nativeGpu = typeof adapter['vendor']==='string' && !software
    const success=smoke['ok']===true && smoke['mode']==='gpu' && nativeGpu &&
      smoke['mismatches']===0 && smoke['frames']===8
    const summary={ok:success, date:new Date().toISOString(), browser:executable,
      test:'visual-swarm WebGPU eight-frame full-wire differential',
      adapter, software, nativeGpu, ...smoke}
    if(process.argv.includes('--screenshot')){
      const screenshot=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})
      const base64=screenshot?.['data']
      if(typeof base64==='string')
        writeFileSync(report.replace(/\.json$/,'.png'), Buffer.from(base64,'base64'))
    }
    return summary
  } finally {socket.close()}
}

try {
  mkdirSync(dirname(report),{recursive:true})
  const result=await run()
  writeFileSync(report,JSON.stringify(result,null,2)+'\n')
  console.log(JSON.stringify(result,null,2))
  if(!result.ok)process.exitCode=1
} catch(error) {
  const result={ok:false,error:String(error),date:new Date().toISOString()}
  mkdirSync(dirname(report),{recursive:true})
  writeFileSync(report,JSON.stringify(result,null,2)+'\n')
  console.error(result)
  process.exitCode=1
} finally {
  browser?.kill()
  try{rmSync(profile,{recursive:true,force:true})}catch{}
  // Bun 1.3.6 on Windows can crash during async teardown of an HTML-route server
  // after the CDP WebSocket closes. Exit explicitly after cleaning up Chrome.
  void server.stop(true)
}
process.exit(process.exitCode ?? 0)

// Hardware-backed differential gate. Uses Chrome/Edge DevTools without a test framework dependency.
// Run: bun scripts/verify-webgpu-browser.ts
import page from '../assets/handwired-webgpu.html'
import {mkdtempSync, rmSync, writeFileSync} from 'node:fs'
import {tmpdir} from 'node:os'
import {join} from 'node:path'

const server = Bun.serve({hostname: '127.0.0.1', port: 39081, routes: {'/': page}})
const report = process.argv[2] ?? 'webgpu-browser-results.json'
const profile = mkdtempSync(join(tmpdir(), 'handwired-chrome-'))
const binaries = process.platform === 'win32' ? [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
] : process.platform === 'darwin' ? [
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
] : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser']
const port = 49381
let child: ReturnType<typeof Bun.spawn> | undefined
const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))
const browserMode = process.argv[3] === 'benchmark' ? 'benchmark' : 'smoke'
const browserUrl = 'http://127.0.0.1:39081/' + (browserMode === 'smoke' ? '?smoke=1' :
  process.argv[4] === undefined ? '' : '?steps=' + encodeURIComponent(process.argv[4]))

async function startBrowser(binary: string) {
  const result = Bun.spawn([
    binary, '--headless=new', '--no-first-run', '--disable-extensions', '--disable-gpu-sandbox',
    ...(process.env['HANDWIRED_GPU_SOFTWARE'] === '1' ? ['--enable-unsafe-swiftshader', '--use-angle=swiftshader'] : []),
    '--enable-unsafe-webgpu', '--enable-webgpu-developer-features',
    '--remote-allow-origins=*',
    '--remote-debugging-port=' + port, '--user-data-dir=' + profile, 'about:blank',
  ], {stdout: 'ignore', stderr: 'ignore'})
  for (let attempt = 0; attempt < 45; attempt++) {
    await delay(250)
    try {
      const response = await fetch('http://127.0.0.1:' + port + '/json/list')
      if (response.ok) {
        const pages = await response.json() as {type: string, webSocketDebuggerUrl: string}[]
        const target = pages.find(page => page.type === 'page')
        if (target) return {result, target}
      }
    } catch {}
    if (result.exitCode !== null) break
  }
  result.kill()
  return null
}

async function verify() {
  let target: {type: string, webSocketDebuggerUrl: string} | undefined
  let browser = ''
  for (const binary of binaries) {
    try {
      const response = await startBrowser(binary)
      if (response) {
        child = response.result
        target = response.target
        browser = binary
        break
      }
    } catch { /* Browser executable unavailable; try next installed browser */ }
  }
  if (!target) throw new Error('No browser exposed its DevTools endpoint')
  const socket = new WebSocket(target.webSocketDebuggerUrl)
  const pending = new Map<number, (value: {result?: Record<string, unknown>, error?: unknown}) => void>()
  let sequence = 0
  socket.onmessage = event => {
    const message = JSON.parse(String(event.data)) as {
      id?: number, result?: Record<string, unknown>, error?: unknown
    }
    if (message.id !== undefined) {
      pending.get(message.id)?.(message)
      pending.delete(message.id)
    }
  }
  await new Promise<void>((resolve, reject) => {
    socket.onopen = () => resolve()
    socket.onerror = () => reject(new Error('DevTools websocket failed'))
  })
  const command = async (method: string, params: Record<string, unknown> = {}) => {
    const id = ++sequence
    const answer = new Promise<{result?: Record<string, unknown>, error?: unknown}>(resolve => pending.set(id, resolve))
    socket.send(JSON.stringify({id, method, params}))
    const result = await answer
    if (result.error !== undefined && result.error !== null) throw new Error(JSON.stringify(result.error))
    return result.result
  }
  await command('Page.enable')
  await command('Runtime.enable')
  await command('Page.navigate', {url: browserUrl})
  for (let attempt = 0; attempt < 90; attempt++) {
    await delay(350)
    const response = await command('Runtime.evaluate', {
      expression: 'JSON.stringify({status:document.body?.dataset.status, smoke:document.querySelector("#smoke")?.textContent})',
      returnByValue: true,
    })
    const textValue = (response?.['result'] as {value?: string} | undefined)?.value
    if (textValue === undefined || textValue.length === 0) continue
    const info = JSON.parse(textValue) as {status?: string, smoke?: string}
    if (info.status === 'pass' || info.status === 'fail') {
      const details = JSON.parse(info.smoke ?? '{}') as Record<string, unknown>
      const reportValue = {
        browser, test: browserMode === 'smoke' ? 'actual WebGPU mixed-operation differential' : 'GPU/CPU visual-swarm benchmark',
        date: new Date().toISOString(), ...details,
      }
      writeFileSync(report, JSON.stringify(reportValue, null, 2) + '\n')
      if (browserMode === 'benchmark' && process.argv.includes('--screenshot')) {
        const capture = await command('Page.captureScreenshot', {format: 'png', captureBeyondViewport: true})
        const base64 = capture?.['data']
        if (typeof base64 === 'string') {
          writeFileSync(report.replace(/\.json$/, '.png'), Buffer.from(base64, 'base64'))
        }
      }
      console.log(JSON.stringify(reportValue, null, 2))
      socket.close()
      if (info.status !== 'pass') process.exitCode = 1
      return
    }
  }
  socket.close()
  throw new Error('Browser page did not complete its hardware test')
}

try {
  await verify()
} catch (error) {
  const details = {ok: false, error: String(error), date: new Date().toISOString()}
  writeFileSync(report, JSON.stringify(details, null, 2) + '\n')
  console.error(details)
  process.exitCode = 1
} finally {
  child?.kill()
  await server.stop()
  try {rmSync(profile, {recursive: true, force: true})} catch { /* Windows browser file locks */ }
}

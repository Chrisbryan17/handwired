import {createWebGpuEvaluator, type WebGpuCircuitEvaluator} from '../../src/runtime/webgpu.ts'
import {buildSwarmCircuit, SOURCE_SIZE} from './model.ts'
import {buildSwarmInputs, initialPopulation, projectState, referenceFrame, type PointerState} from './simulation.ts'

const query = new URLSearchParams(location.search)
const smoke = query.get('smoke') === '1'
const countSelect = document.querySelector<HTMLSelectElement>('#count')!
const strategySelect = document.querySelector<HTMLSelectElement>('#strategy')!
const pointerSelect = document.querySelector<HTMLSelectElement>('#pointer')!
const pauseButton = document.querySelector<HTMLButtonElement>('#pause')!
const restartButton = document.querySelector<HTMLButtonElement>('#restart')!
const status = document.querySelector<HTMLElement>('#status')!
const cpuCanvas = document.querySelector<HTMLCanvasElement>('#cpu')!
const gpuCanvas = document.querySelector<HTMLCanvasElement>('#gpu')!
const cpuContext = cpuCanvas.getContext('2d')!
const gpuContext = gpuCanvas.getContext('2d')!
const {circuit, outputWires} = buildSwarmCircuit()
const width = circuit.sourceSize + circuit.operations.length
const put = (id: string, value: string) => {document.getElementById(id)!.textContent = value}
put('operations', String(circuit.operations.length))
let count = smoke ? Number(query.get('count') ?? '128') : 512
if (!Number.isSafeInteger(count) || count < 1 || count > 8192) count = 128
countSelect.value = String(count)
let species = initialPopulation(count).species
let cpuState = initialPopulation(count).state
let gpuState = cpuState.slice()
let backend: WebGpuCircuitEvaluator | null = null
let requestedRestart = true, paused = false, frames = 0, lastTime = performance.now()
let maxError = 0, mismatches = 0, totalChecked = 0
let cpuMs = 0, gpuMs = 0
const pointer: PointerState = {x: 0.5, y: 0.5, mode: 'off'}
const colors = ['#74ffdc', '#f5ba84', '#a9a3ff']
const radiance = ['#74ffdc44', '#f5ba8444', '#a9a3ff44']

function configureCanvas(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D) {
  const rect = canvas.getBoundingClientRect()
  const pixelRatio = Math.min(2, window.devicePixelRatio || 1)
  const pxWidth = Math.max(1, Math.round(rect.width * pixelRatio))
  const pxHeight = Math.max(1, Math.round(rect.height * pixelRatio))
  if (canvas.width !== pxWidth || canvas.height !== pxHeight) {
    canvas.width = pxWidth
    canvas.height = pxHeight
  }
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
  return {width: rect.width, height: rect.height}
}

function render(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, state: Float32Array) {
  const rect = configureCanvas(canvas, context)
  context.fillStyle = 'rgba(7,12,27,0.29)'
  context.fillRect(0, 0, rect.width, rect.height)
  for (let i = 0; i < count; i++) {
    const at = i * 4
    const x = state[at]! * rect.width, y = state[at + 1]! * rect.height
    const vx = state[at + 2]!, vy = state[at + 3]!
    const angle = Math.atan2(vy, vx)
    const headX = Math.cos(angle), headY = Math.sin(angle)
    const size = 3 + Math.min(2.5, Math.hypot(vx, vy) * 3)
    const sideX = -headY, sideY = headX, s = species[i]!
    context.strokeStyle = radiance[s]!
    context.lineWidth = 2.4
    context.beginPath()
    context.moveTo(x - headX * size * 2, y - headY * size * 2)
    context.lineTo(x, y)
    context.stroke()
    context.fillStyle = colors[s]!
    context.beginPath()
    context.moveTo(x + headX * size, y + headY * size)
    context.lineTo(x - headX * size * 0.8 + sideX * size * 0.45,
      y - headY * size * 0.8 + sideY * size * 0.45)
    context.lineTo(x - headX * size * 0.8 - sideX * size * 0.45,
      y - headY * size * 0.8 - sideY * size * 0.45)
    context.closePath()
    context.fill()
  }
  if (pointer.mode !== 'off') {
    const x = pointer.x * rect.width, y = pointer.y * rect.height
    context.strokeStyle = pointer.mode === 'attract' ? '#84ffe0a0' : '#ff9a94aa'
    context.lineWidth = 1
    context.beginPath()
    context.arc(x, y, 18, 0, Math.PI * 2)
    context.stroke()
    context.beginPath()
    context.arc(x, y, 3, 0, Math.PI * 2)
    context.fillStyle = context.strokeStyle
    context.fill()
  }
}

async function restart() {
  backend?.destroy()
  backend = null
  count = smoke ? count : Number(countSelect.value)
  const population = initialPopulation(count, 3807)
  species = population.species
  cpuState = population.state
  gpuState = cpuState.slice()
  const strategy = strategySelect.value as 'specialized' | 'monolithic' | 'layers'
  status.textContent = 'Compiling ' + strategy + ' Handwired circuit…'
  put('device', 'INITIALIZING')
  backend = await createWebGpuEvaluator(circuit, {
    mode: smoke ? 'gpu' : 'auto', capacity: count, strategy,
  })
  put('device', backend.mode === 'gpu' ? 'LIVE WEBGPU' : 'CPU FALLBACK')
  status.textContent = backend.mode === 'gpu'
    ? 'WebGPU executing the steering circuit. Point into either flock to interact.'
    : 'WebGPU unavailable. Showing deterministic CPU fallback. ' + (backend.fallbackReason ?? '')
  maxError = 0; mismatches = 0; totalChecked = 0; frames = 0
  for (const canvas of [cpuCanvas, gpuCanvas]) {
    const ctx = canvas.getContext('2d')!
    const size = configureCanvas(canvas, ctx)
    ctx.fillStyle = '#070c1b'
    ctx.fillRect(0, 0, size.width, size.height)
  }
  requestedRestart = false
}

function compareAgainstCpuOracle(input: Float32Array, actual: Float32Array) {
  // Check every source and output wire on the exact data the GPU received.
  for (let i = 0; i < count; i++) {
    const expected = circuit.evaluateReference(input.subarray(i * SOURCE_SIZE, (i + 1) * SOURCE_SIZE))
    for (let j = 0; j < width; j++) {
      const delta = Math.abs(expected[j]! - actual[i * width + j]!)
      if (!Number.isFinite(delta) || delta > 0.00002) mismatches++
      maxError = Math.max(maxError, delta)
      totalChecked++
    }
  }
}

async function tick() {
  if (requestedRestart || backend === null) await restart()
  if (paused) return
  const startCpu = performance.now()
  cpuState = referenceFrame(cpuState, species, pointer).state
  cpuMs = performance.now() - startCpu
  const gpuInput = buildSwarmInputs(gpuState, species, pointer)
  const startGpu = performance.now()
  const output = await backend!.evaluate(gpuInput)
  gpuMs = performance.now() - startGpu
  if (smoke || frames % 5 === 0) compareAgainstCpuOracle(gpuInput, output)
  gpuState = projectState(output, count, width, outputWires)
  render(cpuCanvas, cpuContext, cpuState)
  render(gpuCanvas, gpuContext, gpuState)
  frames++
  if (frames % 6 === 0 || smoke) {
    const now = performance.now()
    const fps = frames === 0 ? 0 : 1000 / Math.max(1, now - lastTime) * Math.min(6, frames)
    put('fps', fps.toFixed(1))
    put('cpu-ms', cpuMs.toFixed(2) + ' ms/frame')
    put('gpu-ms', gpuMs.toFixed(2) + ' ms/frame')
    put('error', maxError.toExponential(2))
    put('wires', totalChecked.toLocaleString())
    put('mismatches', mismatches.toLocaleString())
    lastTime = now
  }
  document.body.dataset.status = mismatches === 0 ? 'pass' : 'fail'
  if (smoke && frames >= 8) {
    const result = {ok: mismatches === 0 && backend!.mode === 'gpu',
      mode: backend!.mode, frames, count, strategy: strategySelect.value,
      circuitOperations: circuit.operations.length, totalChecked, mismatches, maxError, cpuMs, gpuMs}
    document.querySelector<HTMLElement>('#smoke')!.textContent = JSON.stringify(result)
    document.body.dataset.status = result.ok ? 'pass' : 'fail'
    paused = true
  }
}

async function loop() {
  try {await tick()}
  catch (error) {
    document.body.dataset.status = 'fail'
    status.textContent = 'Experiment stopped: ' + String(error)
    const result = {ok: false, reason: String(error), frames}
    document.querySelector<HTMLElement>('#smoke')!.textContent = JSON.stringify(result)
    paused = true
  }
  if (!(smoke && paused)) requestAnimationFrame(() => {void loop()})
}

for (const canvas of [cpuCanvas, gpuCanvas]) {
  canvas.addEventListener('pointermove', event => {
    const rect = canvas.getBoundingClientRect()
    pointer.x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
    pointer.y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))
    pointer.mode = pointerSelect.value as PointerState['mode']
  })
  canvas.addEventListener('pointerleave', () => {pointer.mode = 'off'})
}
pointerSelect.addEventListener('change', () => {pointer.mode = 'off'})
pauseButton.addEventListener('click', () => {
  paused = !paused
  pauseButton.textContent = paused ? 'RESUME' : 'PAUSE'
})
restartButton.addEventListener('click', () => {requestedRestart = true; paused = false; pauseButton.textContent = 'PAUSE'})
countSelect.addEventListener('change', () => {requestedRestart = true})
strategySelect.addEventListener('change', () => {requestedRestart = true})
void loop()

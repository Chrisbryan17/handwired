import {Circuit} from '../src/runtime/numeric-ui-core.ts'
import {createWebGpuEvaluator} from '../src/runtime/webgpu.ts'

const count = 8192
const canvas = document.querySelector<HTMLCanvasElement>('#swarm')!
const context = canvas.getContext('2d')!
const status = document.querySelector<HTMLElement>('#status')!
const results = document.querySelector<HTMLElement>('#results')!
const runButton = document.querySelector<HTMLButtonElement>('#run')!
const smoke = document.querySelector<HTMLElement>('#smoke')!
const circuit = new Circuit(4)
const x = circuit.source(0), y = circuit.source(1)
const vx = circuit.source(2), vy = circuit.source(3)
const xNext = circuit.roundAffine([[x, 1], [vx, 0.016]])
const yNext = circuit.roundAffine([[y, 1], [vy, 0.016]])
const forceX = circuit.roundAffine([[xNext, -0.07]], 0.035)
const forceY = circuit.roundAffine([[yNext, -0.07]], 0.035)
const vxNext = circuit.roundAffine([[vx, 0.997], [forceX, 0.016]])
const vyNext = circuit.roundAffine([[vy, 0.997], [forceY, 0.016]])
const state = [
  {source: 0, from: xNext}, {source: 1, from: yNext},
  {source: 2, from: vxNext}, {source: 3, from: vyNext},
]
const width = circuit.sourceSize + circuit.operations.length
const sources = new Float32Array(count * circuit.sourceSize)
let seed = 27
const random = () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) | 0
  return (seed >>> 0) / 4294967296
}
for (let i = 0; i < count; i++) {
  const theta = random() * 6.283185307179586
  const r = Math.sqrt(random()) * 0.36
  sources.set([0.5 + Math.cos(theta) * r, 0.5 + Math.sin(theta) * r,
    (random() - 0.5) * 0.32, (random() - 0.5) * 0.32], i * 4)
}

function draw(cpu: Float32Array, gpu: Float32Array) {
  context.fillStyle = '#091326'
  context.fillRect(0, 0, canvas.width, canvas.height)
  for (const [values, left, color] of [[cpu, 0, '#51dcdf'], [gpu, canvas.width / 2, '#e5b8ff']] as const) {
    context.fillStyle = color
    for (let i = 0; i < count; i++) {
      const at = i * width
      const px = left + 18 + ((values[at + xNext] ?? 0) * (canvas.width / 2 - 36))
      const py = 14 + ((values[at + yNext] ?? 0) * (canvas.height - 28))
      context.fillRect(px, py, 1.5, 1.5)
    }
  }
  context.fillStyle = '#e8f0ff'
  context.font = '600 17px system-ui'
  context.fillText('CPU Reference', 20, 30)
  context.fillText('WebGPU Circuit', canvas.width / 2 + 20, 30)
}

// Independent, hand-written implementation of the same six-operation circuit.
// Its FP32 steps intentionally mirror the documented reference arithmetic.
function handwrittenJavaScript(input: Float32Array, steps: number): Float32Array {
  const output = new Float32Array(count * width)
  const f = Math.fround
  const timeStep = f(0.016), drag = f(0.997), stiffness = f(-0.07), center = f(0.035)
  for (let i = 0; i < count; i++) {
    const at = i * 4
    let px = input[at]!, py = input[at + 1]!
    let velocityX = input[at + 2]!, velocityY = input[at + 3]!
    for (let step = 0; step < steps; step++) {
      const xValue = f(f(px) + f(timeStep * velocityX))
      const yValue = f(f(py) + f(timeStep * velocityY))
      const forceXValue = f(center + f(stiffness * xValue))
      const forceYValue = f(center + f(stiffness * yValue))
      const vxValue = f(f(drag * velocityX) + f(timeStep * forceXValue))
      const vyValue = f(f(drag * velocityY) + f(timeStep * forceYValue))
      if (step === steps - 1) {
        const offset = i * width
        output.set([px, py, velocityX, velocityY,
          xValue, yValue, forceXValue, forceYValue, vxValue, vyValue], offset)
      }
      px = xValue
      py = yValue
      velocityX = vxValue
      velocityY = vyValue
    }
  }
  return output
}

async function compare() {
  runButton.disabled = true
  status.textContent = 'Running independent CPU and WebGPU executions...'
  const steps = Number(new URLSearchParams(location.search).get('steps')
    ?? document.querySelector<HTMLSelectElement>('#steps')!.value)
  const cpu = await createWebGpuEvaluator(circuit, {mode: 'cpu', capacity: count})
  const beganCpu = performance.now()
  const cpuValues = await cpu.evaluate(sources, {steps, stateUpdates: state})
  const cpuMs = performance.now() - beganCpu
  cpu.destroy()
  const beganJs = performance.now()
  const jsValues = handwrittenJavaScript(sources, steps)
  const jsMs = performance.now() - beganJs

  const startedGpuInit = performance.now()
  const gpu = await createWebGpuEvaluator(circuit, {mode: 'gpu', capacity: count})
  const gpuInitMs = performance.now() - startedGpuInit
  const beganGpu = performance.now()
  const gpuValues = await gpu.evaluate(sources, {steps, stateUpdates: state})
  const gpuMs = performance.now() - beganGpu
  gpu.destroy()
  const startedLayerInit = performance.now()
  const layered = await createWebGpuEvaluator(circuit, {mode: 'gpu', capacity: count, strategy: 'layers'})
  const layeredInitMs = performance.now() - startedLayerInit
  const beganLayered = performance.now()
  const layeredValues = await layered.evaluate(sources, {steps, stateUpdates: state})
  const layeredMs = performance.now() - beganLayered
  const layerCount = layered.layerCount
  layered.destroy()
  let mismatches = 0, maxDifference = 0, jsMismatches = 0
  for (let i = 0; i < gpuValues.length; i++) {
    const difference = Math.abs(cpuValues[i]! - gpuValues[i]!)
    const layerDifference = Math.abs(cpuValues[i]! - layeredValues[i]!)
    const jsDifference = Math.abs(cpuValues[i]! - jsValues[i]!)
    if (jsDifference > 0.00002) jsMismatches++
    if (difference > 0.00002 || layerDifference > 0.00002) mismatches++
    maxDifference = Math.max(maxDifference, difference, layerDifference)
  }
  draw(cpuValues, gpuValues)
  results.textContent = 'Particles: ' + count.toLocaleString()
    + ' · Steps: ' + steps + ' · CPU: ' + cpuMs.toFixed(2) + ' ms'
    + ' · Handwritten JS: ' + jsMs.toFixed(2) + ' ms'
    + ' · GPU (including upload + readback): ' + gpuMs.toFixed(2) + ' ms'
    + ' · Layers: ' + layeredMs.toFixed(2) + ' ms (' + layerCount + ' layers)'
    + ' · Monolithic speedup: ' + (cpuMs / gpuMs).toFixed(2) + '×'
    + ' · GPU mismatches (>2e-5): ' + mismatches
    + ' · JS mismatches: ' + jsMismatches
    + ' · Maximum error: ' + maxDifference.toExponential(2)
  status.textContent = 'Both sides use exactly the same Handwired circuit.'
  smoke.textContent = JSON.stringify({ok: mismatches === 0, samples: count, steps,
    cpuMs, jsMs, gpuMs, layeredMs, gpuInitMs, layeredInitMs, layerCount, speedup: cpuMs / gpuMs,
    layeredSpeedup: cpuMs / layeredMs, mismatches, jsMismatches, maxDifference})
  document.body.dataset.status = mismatches === 0 && jsMismatches === 0 ? 'pass' : 'fail'
  runButton.disabled = false
}

async function smokeTest() {
  document.body.dataset.debug = 'true'
  const test = new Circuit(5)
  const a = test.source(0), b = test.source(1)
  const gate = test.source(2), index = test.source(3), score = test.source(4)
  test.assumeRange(gate, 0, 1)
  test.assumeRange(index, -2, 5)
  test.assumeRange(score, 0, 1)
  const sum = test.roundAffine([[a, 0.1], [b, 0.2]], 0.3)
  const relu = test.reglu([[sum, 1]], 0, [[gate, 1]], 0)
  const bit = test.boolean(score)
  const read = test.indexReadAffine([[index, 1]], 0, [a, b, relu, bit], 'zero', 'clamp')
  test.roundAffine([[read, 1], [sum, -1]], 2)
  const data = new Float32Array(512 * 5)
  for (let i = 0; i < 512; i++) {
    data.set([i / 12 - 2, 2 - i / 19, i % 3 / 2, (i % 8) - 2, (i % 7 === 0 ? 0.5 : i % 4 / 3)], i * 5)
  }
  const adapter = await navigator.gpu?.requestAdapter()
  if (!adapter) throw new Error('No WebGPU adapter')
  const device = await adapter.requestDevice()
  const backend = await createWebGpuEvaluator(test, {mode: 'gpu', device, capacity: 512})
  let mismatches = 0, maxError = 0
  let stateMismatches = 0, stateMaxError = 0
  try {
    const values = await backend.evaluate(data)
    const countPerSample = test.sourceSize + test.operations.length
    for (let i = 0; i < 512; i++) {
      const expected = test.evaluateReference(data.subarray(i * 5, (i + 1) * 5))
      for (let w = 0; w < expected.length; w++) {
        const actual = values[i * countPerSample + w]!
        const err = Math.abs(actual - expected[w]!)
        maxError = Math.max(maxError, err)
        if (!Number.isFinite(actual) || err > 0.00002) mismatches++
      }
    }
    backend.load(data)
    await backend.run(5, [{source: 0, from: sum}])
    const stepped = await backend.readValues()
    for (let i = 0; i < 512; i++) {
      const source = data.slice(i * 5, (i + 1) * 5)
      let reference = test.evaluateReference(source)
      for (let step = 1; step < 5; step++) {
        source[0] = reference[sum]!
        reference = test.evaluateReference(source)
      }
      for (let w = 0; w < reference.length; w++) {
        const actual = stepped[i * countPerSample + w]!
        const err = Math.abs(actual - reference[w]!)
        stateMaxError = Math.max(stateMaxError, err)
        if (!Number.isFinite(actual) || err > 0.00002) stateMismatches++
      }
    }
  } finally {
    backend.destroy()
  }
  let layerMismatches = 0, layerStateMismatches = 0
  const layerBackend = await createWebGpuEvaluator(test, {mode: 'gpu', device, capacity: 512, strategy: 'layers'})
  try {
    const layerValues = await layerBackend.evaluate(data)
    const width = test.sourceSize + test.operations.length
    for (let i = 0; i < 512; i++) {
      const reference = test.evaluateReference(data.subarray(i * 5, (i + 1) * 5))
      for (let w = 0; w < width; w++) {
        if (!Number.isFinite(layerValues[i * width + w]!) ||
          Math.abs(layerValues[i * width + w]! - reference[w]!) > 0.00002) layerMismatches++
      }
    }
    layerBackend.load(data)
    await layerBackend.run(5, [{source: 0, from: sum}])
    const stepped = await layerBackend.readValues()
    for (let i = 0; i < 512; i++) {
      const source = data.slice(i * 5, (i + 1) * 5)
      let reference = test.evaluateReference(source)
      for (let step = 1; step < 5; step++) {
        source[0] = reference[sum]!
        reference = test.evaluateReference(source)
      }
      for (let w = 0; w < width; w++) {
        if (!Number.isFinite(stepped[i * width + w]!) ||
          Math.abs(stepped[i * width + w]! - reference[w]!) > 0.00002) layerStateMismatches++
      }
    }
  } finally {
    layerBackend.destroy()
  }
  // Integrate with an unmodified, actual HandwiredOS application circuit:
  // 800 operations in 67 dependency layers at the checked-in baseline.
  const {createValueListCircuit} = await import('../src/handwiredos/value-list.ts')
  const integrated = createValueListCircuit()
  const integratedSource = Float32Array.from(integrated.initialSource)
  const integratedExpected = integrated.circuit.evaluateReference(integratedSource)
  let integratedMismatches = 0, integratedBitMismatches = 0, integratedMaxError = 0
  let integratedLayerCount = 0
  for (const strategy of ['monolithic', 'layers'] as const) {
    const realEvaluator = await createWebGpuEvaluator(integrated.circuit, {
      mode: 'gpu', device, capacity: 1, strategy,
    })
    try {
      integratedLayerCount = realEvaluator.layerCount
      const actual = await realEvaluator.evaluate(integratedSource)
      for (let i = 0; i < integratedExpected.length; i++) {
        const expected = integratedExpected[i]!, got = actual[i]!
        if (!Object.is(expected, got)) integratedBitMismatches++
        const difference = Math.abs(expected - got)
        if (Number.isFinite(difference)) integratedMaxError = Math.max(integratedMaxError, difference)
        if (!(Object.is(expected, got) ||
          (Number.isFinite(difference) && difference <= 0.0001))) integratedMismatches++
      }
    } finally {
      realEvaluator.destroy()
    }
  }
  // Second unmodified application: photo gallery, including its boolean atoms.
  const {createPhotoGalleryCircuit} = await import('../src/handwiredos/photo-gallery.ts')
  const gallery = createPhotoGalleryCircuit()
  const gallerySource = Float32Array.from(gallery.initialSource)
  const galleryExpected = gallery.circuit.evaluateReference(gallerySource)
  let galleryMismatches = 0, galleryBitMismatches = 0, galleryMaxError = 0
  let galleryLayerCount = 0
  for (const strategy of ['monolithic', 'layers'] as const) {
    const galleryEvaluator = await createWebGpuEvaluator(gallery.circuit, {
      mode: 'gpu', device, capacity: 1, strategy,
    })
    try {
      galleryLayerCount = galleryEvaluator.layerCount
      const actual = await galleryEvaluator.evaluate(gallerySource)
      for (let i = 0; i < galleryExpected.length; i++) {
        const expected = galleryExpected[i]!, got = actual[i]!
        if (!Object.is(expected, got)) galleryBitMismatches++
        const difference = Math.abs(expected - got)
        if (Number.isFinite(difference)) galleryMaxError = Math.max(galleryMaxError, difference)
        if (!(Object.is(expected, got) ||
          (Number.isFinite(difference) && difference <= 0.0001))) galleryMismatches++
      }
    } finally {
      galleryEvaluator.destroy()
    }
  }
  const doomed = await createWebGpuEvaluator(test, {mode: 'gpu', device, capacity: 512})
  doomed.load(data)
  device.destroy()
  await device.lost
  let deviceLossCaught = false
  try {
    await doomed.run()
  } catch (error) {
    deviceLossCaught = String(error).includes('device lost')
  } finally {
    doomed.destroy()
  }
  const info = {ok: mismatches === 0 && stateMismatches === 0 && layerMismatches === 0
    && layerStateMismatches === 0 && integratedMismatches === 0 && galleryMismatches === 0
    && deviceLossCaught,
    deviceLossCaught, galleryOperations: gallery.circuit.operations.length,
    galleryLayerCount, galleryMismatches, galleryBitMismatches, galleryMaxError,
    integratedOperations: integrated.circuit.operations.length,
    integratedLayerCount, integratedMismatches, integratedBitMismatches, integratedMaxError,
    mismatches, maxError, stateMismatches, stateMaxError, layerMismatches, layerStateMismatches,
    adapter: adapter.info?.description || adapter.info?.vendor || 'WebGPU adapter',
    testedSamples: 512, operationKinds: ['affine', 'reglu', 'boolean', 'index']}
  smoke.textContent = JSON.stringify(info)
  document.body.dataset.status = info.ok ? 'pass' : 'fail'
  if (!info.ok) throw new Error('differential mismatches: ' + mismatches)
}

runButton.addEventListener('click', () => {
  void compare().catch(error => {
    status.textContent = String(error)
    runButton.disabled = false
  })
})
if (new URLSearchParams(location.search).has('smoke')) {
  void smokeTest().catch(error => {
    smoke.textContent = JSON.stringify({ok: false, error: String(error)})
    document.body.dataset.status = 'fail'
  })
} else {
  void compare().catch(error => {status.textContent = String(error); runButton.disabled = false})
}

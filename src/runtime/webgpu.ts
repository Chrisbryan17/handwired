// General-purpose batched WebGPU execution for Handwired circuits.
// This is an interpreter kernel over serialized operations, not JSON-search-specific code.
import {type Circuit, invariant, type Wire} from './numeric-ui-core.ts'
import {circuitWgsl} from './webgpu-shader.ts'

const Usage = {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128} as const
const MapMode = {READ: 1} as const
const operationStride = 48
const termStride = 8
const workgroupSize = 64

export type StateUpdate = {source: number, from: Wire}
export type WebGpuOptions = {
  device?: GPUDevice
  mode?: 'auto' | 'gpu' | 'cpu'
  capacity?: number
}

type PackedCircuit = {
  operations: Uint8Array
  terms: Uint8Array
  candidates: Uint32Array
}

function serialize(circuit: Circuit): PackedCircuit {
  const operations = new ArrayBuffer(Math.max(operationStride, circuit.operations.length * operationStride))
  const opView = new DataView(operations)
  const terms: [number, number][] = []
  const candidates: number[] = []
  for (let index = 0; index < circuit.operations.length; index++) {
    const operation = circuit.operations[index]!
    const start = index * operationStride
    const leftStart = terms.length
    for (const [wire, coefficient] of operation.leftTerms) terms.push([wire, Math.fround(coefficient)])
    const gateStart = terms.length
    for (const [wire, coefficient] of operation.gateTerms) terms.push([wire, Math.fround(coefficient)])
    const candidateStart = candidates.length
    const table = operation.indexRead?.table.candidates ?? []
    for (const wire of table) candidates.push(wire)
    const kind = {affine: 0, reglu: 1, boolean: 2, index: 3}[operation.kind]
    const fields: readonly (readonly [number, number])[] = [
      [0, kind], [4, leftStart], [8, operation.leftTerms.length],
      [12, gateStart], [16, operation.gateTerms.length],
      [20, candidateStart], [24, table.length],
      [28, (operation.indexRead?.below === 'zero' ? 1 : 0)
        | (operation.indexRead?.above === 'zero' ? 2 : 0)],
    ]
    for (const [offset, value] of fields) opView.setUint32(start + offset, value, true)
    opView.setFloat32(start + 32, operation.leftBias, true)
    opView.setFloat32(start + 36, operation.gateBias, true)
  }
  const termBytes = new ArrayBuffer(Math.max(termStride, terms.length * termStride))
  const termView = new DataView(termBytes)
  for (let index = 0; index < terms.length; index++) {
    termView.setUint32(index * termStride, terms[index]![0], true)
    termView.setFloat32(index * termStride + 4, terms[index]![1], true)
  }
  return {
    operations: new Uint8Array(operations),
    terms: new Uint8Array(termBytes),
    candidates: Uint32Array.from(candidates.length ? candidates : [0]),
  }
}

export class WebGpuCircuitEvaluator {
  readonly mode: 'gpu' | 'cpu'
  readonly capacity: number
  readonly sourceSize: number
  readonly width: number
  private count = 0
  private hasRun = false
  private disposed = false
  private loss: string | null = null
  private readonly buffers: GPUBuffer[] = []
  private sourceBuffer?: GPUBuffer
  private valueBuffer?: GPUBuffer
  private stateBuffer?: GPUBuffer
  private parameterBuffer?: GPUBuffer
  private bindGroup?: GPUBindGroup
  private executePipeline?: GPUComputePipeline
  private advancePipeline?: GPUComputePipeline
  private sources: Float32Array
  private values: Float32Array

  private constructor(private readonly circuit: Circuit,
    private readonly device: GPUDevice | undefined, capacity: number) {
    this.capacity = capacity
    this.mode = device ? 'gpu' : 'cpu'
    this.sourceSize = circuit.sourceSize
    this.width = circuit.sourceSize + circuit.operations.length
    this.sources = new Float32Array(capacity * this.sourceSize)
    this.values = new Float32Array(capacity * this.width)
  }

  static async create(circuit: Circuit, options: WebGpuOptions = {}): Promise<WebGpuCircuitEvaluator> {
    const capacity = options.capacity ?? 256
    invariant(Number.isSafeInteger(capacity) && capacity > 0, 'capacity must be a positive integer')
    const width = circuit.sourceSize + circuit.operations.length
    invariant(width > 0, 'cannot execute an empty circuit')
    invariant(width * capacity <= 0x3fffffff, 'batch exceeds supported buffer size')
    let device = options.device
    if (options.mode !== 'cpu' && device === undefined) {
      const gpu = typeof navigator !== 'undefined' ? navigator.gpu : undefined
      const adapter = gpu === undefined ? null : await gpu.requestAdapter()
      device = (await adapter?.requestDevice()) ?? undefined
    }
    if (!device && options.mode === 'gpu') throw new Error('WebGPU adapter/device unavailable')
    const evaluator = new WebGpuCircuitEvaluator(circuit, device, capacity)
    if (device) await evaluator.initializeGpu(device)
    return evaluator
  }

  private buffer(bytes: number, usage: number): GPUBuffer {
    const device = this.device!
    const buffer = device.createBuffer({size: Math.max(4, bytes), usage})
    this.buffers.push(buffer)
    return buffer
  }

  private async initializeGpu(device: GPUDevice): Promise<void> {
    const packed = serialize(this.circuit)
    const sourceBytes = this.capacity * this.sourceSize * 4
    const valueBytes = this.capacity * this.width * 4
    for (const bytes of [sourceBytes, valueBytes, packed.operations.byteLength,
      packed.terms.byteLength, packed.candidates.byteLength, this.sourceSize * 4]) {
      if (Math.max(4, bytes) > device.limits.maxStorageBufferBindingSize ||
        Math.max(4, bytes) > device.limits.maxBufferSize) {
        throw new Error('GPU storage binding limit exceeded; use a smaller capacity')
      }
    }
    this.parameterBuffer = this.buffer(16, Usage.UNIFORM | Usage.COPY_DST)
    this.sourceBuffer = this.buffer(sourceBytes, Usage.STORAGE | Usage.COPY_DST)
    this.valueBuffer = this.buffer(valueBytes, Usage.STORAGE | Usage.COPY_SRC)
    const operationBuffer = this.buffer(packed.operations.byteLength, Usage.STORAGE | Usage.COPY_DST)
    const termBuffer = this.buffer(packed.terms.byteLength, Usage.STORAGE | Usage.COPY_DST)
    const candidateBuffer = this.buffer(packed.candidates.byteLength, Usage.STORAGE | Usage.COPY_DST)
    this.stateBuffer = this.buffer(this.sourceSize * 4, Usage.STORAGE | Usage.COPY_DST)
    const gpuBuffers = [this.parameterBuffer, this.sourceBuffer, this.valueBuffer,
      operationBuffer, termBuffer, candidateBuffer, this.stateBuffer]
    device.queue.writeBuffer(operationBuffer, 0, packed.operations)
    device.queue.writeBuffer(termBuffer, 0, packed.terms)
    device.queue.writeBuffer(candidateBuffer, 0, packed.candidates)
    const module = device.createShaderModule({code: circuitWgsl, label: 'Handwired operation interpreter'})
    const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error')
    if (messages.length) throw new Error('WGSL compilation failed: ' +
      messages.map(message => message.lineNum + ': ' + message.message).join('; '))
    const layout = device.createBindGroupLayout({entries: gpuBuffers.map((_, binding) => ({
      binding, visibility: 4, // GPUShaderStage.COMPUTE
      buffer: {type: binding === 0 ? 'uniform' : binding <= 2 ? 'storage' : 'read-only-storage'},
    }))})
    const pipelineLayout = device.createPipelineLayout({bindGroupLayouts: [layout]})
    this.executePipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout, compute: {module, entryPoint: 'execute'},
    })
    this.advancePipeline = await device.createComputePipelineAsync({
      layout: pipelineLayout, compute: {module, entryPoint: 'advanceState'},
    })
    this.bindGroup = device.createBindGroup({
      layout,
      entries: gpuBuffers.map((buffer, binding) => ({binding, resource: {buffer}})),
    })
    void device.lost.then(info => {this.loss = info.message.length > 0 ? info.message : info.reason})
  }

  private ensureActive(): void {
    if (this.disposed) throw new Error('WebGPU evaluator already destroyed')
    if (this.loss !== null) throw new Error('WebGPU device lost: ' + this.loss)
  }

  load(input: Float32Array, batchSize?: number): void {
    this.ensureActive()
    const count = batchSize ?? (this.sourceSize === 0 ? 0 : input.length / this.sourceSize)
    invariant(Number.isSafeInteger(count) && count > 0 && count <= this.capacity,
      'batch size must be between 1 and capacity')
    invariant(input.length === count * this.sourceSize, 'source table batch has wrong length')
    for (let sample = 0; sample < count; sample++) {
      this.circuit.validateSourceRanges(
        input.subarray(sample * this.sourceSize, (sample + 1) * this.sourceSize))
    }
    this.count = count
    this.hasRun = false
    if (this.device) {
      if (input.byteLength) this.device.queue.writeBuffer(this.sourceBuffer!, 0, input)
      this.device.queue.writeBuffer(this.parameterBuffer!, 0,
        new Uint32Array([count, this.sourceSize, this.width, this.circuit.operations.length]))
    } else {
      this.sources.set(input)
    }
  }

  async run(steps = 1, stateUpdates: readonly StateUpdate[] = []): Promise<void> {
    this.ensureActive()
    invariant(this.count > 0, 'call load before run')
    invariant(Number.isSafeInteger(steps) && steps > 0, 'steps must be a positive integer')
    const wires = Uint32Array.from({length: this.sourceSize}, (_, i) => i)
    const seen = new Set<number>()
    for (const update of stateUpdates) {
      invariant(Number.isInteger(update.source) && update.source >= 0 &&
        update.source < this.sourceSize, 'state source out of bounds')
      invariant(Number.isInteger(update.from) && update.from >= 0 &&
        update.from < this.width, 'state output wire out of bounds')
      invariant(!seen.has(update.source), 'duplicate state update source')
      seen.add(update.source)
      wires[update.source] = update.from
    }
    if (this.device) {
      if (this.sourceSize) this.device.queue.writeBuffer(this.stateBuffer!, 0, wires)
      const encoder = this.device.createCommandEncoder()
      for (let iteration = 0; iteration < steps; iteration++) {
        const pass = encoder.beginComputePass()
        pass.setPipeline(this.executePipeline!)
        pass.setBindGroup(0, this.bindGroup!)
        pass.dispatchWorkgroups(Math.ceil(this.count / workgroupSize))
        pass.end()
        if (stateUpdates.length > 0) {
          const statePass = encoder.beginComputePass()
          statePass.setPipeline(this.advancePipeline!)
          statePass.setBindGroup(0, this.bindGroup!)
          statePass.dispatchWorkgroups(Math.ceil(this.count / workgroupSize))
          statePass.end()
        }
      }
      this.device.queue.submit([encoder.finish()])
      await this.device.queue.onSubmittedWorkDone()
      this.ensureActive()
    } else {
      for (let iteration = 0; iteration < steps; iteration++) {
        for (let sample = 0; sample < this.count; sample++) {
          const start = sample * this.sourceSize
          const results = this.circuit.evaluateReference(
            this.sources.slice(start, start + this.sourceSize))
          this.values.set(results, sample * this.width)
          if (stateUpdates.length > 0) {
            for (let i = 0; i < this.sourceSize; i++) {
              this.sources[start + i] = results[wires[i]!]!
            }
          }
        }
      }
    }
    this.hasRun = true
  }

  async readValues(): Promise<Float32Array> {
    this.ensureActive()
    invariant(this.hasRun, 'call run before readValues')
    if (!this.device) return this.values.slice(0, this.count * this.width)
    const bytes = this.count * this.width * 4
    const staging = this.device.createBuffer({
      size: bytes, usage: Usage.MAP_READ | Usage.COPY_DST,
    })
    try {
      const encoder = this.device.createCommandEncoder()
      encoder.copyBufferToBuffer(this.valueBuffer!, 0, staging, 0, bytes)
      this.device.queue.submit([encoder.finish()])
      await staging.mapAsync(MapMode.READ)
      const output = new Float32Array(staging.getMappedRange().slice(0))
      staging.unmap()
      return output
    } finally {
      staging.destroy()
    }
  }

  async evaluate(input: Float32Array, options: {
    batchSize?: number, steps?: number, stateUpdates?: readonly StateUpdate[]
  } = {}): Promise<Float32Array> {
    this.load(input, options.batchSize)
    await this.run(options.steps ?? 1, options.stateUpdates)
    return this.readValues()
  }

  destroy(): void {
    if (this.disposed) return
    this.disposed = true
    for (const buffer of this.buffers) buffer.destroy()
    this.buffers.length = 0
  }
}

export const createWebGpuEvaluator = (circuit: Circuit, options: WebGpuOptions = {}) => WebGpuCircuitEvaluator.create(circuit, options)

/**
 * Strict WebGPU acquisition for research demos. This module never pretends a
 * software adapter, CPU path or lack of WebGPU is a hardware verification pass.
 * Adapter identification is observational; hardware acceptance requires a
 * browser-specific adapter identity and an actual completed compute probe.
 */
export type AdapterIdentity = {
  vendor: string
  architecture: string
  description: string
  device: string
  suspectedSoftware: boolean
  name: string
}
export type GpuAllocation = {
  device: GPUDevice
  adapter: AdapterIdentity
  bytesAllocated: () => number
  allocate: (bytes: number, usage: GPUBufferUsageFlags, label: string) => GPUBuffer
  check: () => void
  dispose: () => void
}
const softwarePattern = /swiftshader|llvmpipe|softpipe|software|microsoft basic render|lavapipe|cpu raster/i
const makeIdentity = (adapter: GPUAdapter): AdapterIdentity => {
  const info = adapter.info
  const vendor = info.vendor
  const architecture = info.architecture
  const description = info.description
  const device = info.device
  const resolvedName = [vendor, architecture, device, description].filter(v => v.length > 0).join(' / ')
  const name = resolvedName.length > 0 ? resolvedName : 'unidentified adapter'
  return {vendor, architecture, description, device, name,
    suspectedSoftware: softwarePattern.test(name)}
}
export async function requestStrictGpu(label = 'Handwired Frontier'): Promise<GpuAllocation> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    throw new Error('WebGPU is unavailable: this demo requires browser GPU compute')
  }
  const adapter = await navigator.gpu.requestAdapter({powerPreference:'high-performance'})
  if (!adapter) throw new Error('No WebGPU adapter available')
  const identity = makeIdentity(adapter)
  if (identity.suspectedSoftware) throw new Error('Software WebGPU adapter rejected: ' + identity.name)
  const device = await adapter.requestDevice({label})
  let lost: string | null = null
  let disposed = false
  let allocated = 0
  const buffers = new Set<GPUBuffer>()
  void device.lost.then(info => {lost = info.reason + ': ' + info.message})
  const check = () => {
    if (disposed) throw new Error('GPU allocation has been disposed')
    if (lost !== null) throw new Error('GPU device lost: ' + lost)
  }
  const allocate = (bytes: number, usage: GPUBufferUsageFlags, resourceLabel: string): GPUBuffer => {
    check()
    if (!Number.isSafeInteger(bytes) || bytes <= 0 || bytes % 4 !== 0 ||
      bytes > device.limits.maxBufferSize) {
      throw new Error('Invalid GPU buffer size: ' + bytes)
    }
    if ((usage & 128) !== 0 && bytes > device.limits.maxStorageBufferBindingSize) {
      throw new Error('Storage buffer exceeds adapter binding limit: ' + bytes)
    }
    const buffer = device.createBuffer({size: bytes, usage, label: resourceLabel})
    buffers.add(buffer)
    allocated += bytes
    return buffer
  }
  return {
    device, adapter: identity,
    bytesAllocated: () => allocated,
    allocate, check,
    dispose: () => {
      if (disposed) return
      disposed = true
      for (const buffer of buffers) buffer.destroy()
      buffers.clear()
      allocated = 0
      device.destroy()
    },
  }
}

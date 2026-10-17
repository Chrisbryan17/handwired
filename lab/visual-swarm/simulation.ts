import {buildSwarmCircuit, SOURCE_SIZE, STATE_SIZE} from './model.ts'

export type PointerState = {x: number, y: number, mode: 'attract' | 'repel' | 'off'}
export type SwarmPopulation = {state: Float32Array, species: Uint8Array}
export const NEIGHBOR_RADIUS = 0.11
const CELL_COUNT = Math.ceil(1 / NEIGHBOR_RADIUS)
const NEIGHBOR_RADIUS_SQUARED = NEIGHBOR_RADIUS * NEIGHBOR_RADIUS

/** Identical deterministic populations for left and right comparison panes. */
export function initialPopulation(count: number, seed = 3807): SwarmPopulation {
  if (!Number.isSafeInteger(count) || count < 1 || count > 8192) {
    throw new Error('population must contain 1..8192 agents')
  }
  const state = new Float32Array(count * STATE_SIZE), species = new Uint8Array(count)
  let current = seed >>> 0
  const random = () => {
    current = (Math.imul(current, 1664525) + 1013904223) >>> 0
    return current / 4294967296
  }
  for (let i = 0; i < count; i++) {
    const group = i % 3
    species[i] = group
    const theta = random() * Math.PI * 2, radius = Math.sqrt(random()) * 0.16
    const cx = [0.28, 0.72, 0.5][group]!
    const cy = [0.35, 0.35, 0.72][group]!
    const x = Math.max(0.025, Math.min(0.975, cx + Math.cos(theta) * radius))
    const y = Math.max(0.025, Math.min(0.975, cy + Math.sin(theta) * radius))
    state.set([x, y, Math.cos(theta + Math.PI / 2) * 0.27,
      Math.sin(theta + Math.PI / 2) * 0.27], i * STATE_SIZE)
  }
  return {state, species}
}

const limit = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/**
 * Host-side uniform spatial hash (3x3 neighboring cells).
 * Signals are explicit sources shared by CPU and GPU Handwired executors.
 * The same species respond to one another; an interactive cursor influences all groups.
 */
export function buildSwarmInputs(
  state: Float32Array, species: Uint8Array, pointer: PointerState,
): Float32Array {
  const count = species.length
  if (state.length !== count * STATE_SIZE) throw new Error('state length mismatch')
  const inputs = new Float32Array(count * SOURCE_SIZE)
  const buckets: number[][] = Array.from({length: CELL_COUNT * CELL_COUNT}, () => [])
  const cell = (x: number) => limit(Math.floor(x / NEIGHBOR_RADIUS), 0, CELL_COUNT - 1)
  for (let i = 0; i < count; i++) {
    buckets[cell(state[i * 4 + 1]!) * CELL_COUNT + cell(state[i * 4]!)]!.push(i)
  }
  for (let i = 0; i < count; i++) {
    const offset = i * 4, dest = i * SOURCE_SIZE
    const x = state[offset]!, y = state[offset + 1]!
    const vx = state[offset + 2]!, vy = state[offset + 3]!
    inputs.set([x, y, vx, vy], dest)
    let separationX = 0, separationY = 0, sumX = 0, sumY = 0
    let sumVx = 0, sumVy = 0, neighbors = 0
    const gx = cell(x), gy = cell(y)
    for (let yy = Math.max(0, gy - 1); yy <= Math.min(CELL_COUNT - 1, gy + 1); yy++) {
      for (let xx = Math.max(0, gx - 1); xx <= Math.min(CELL_COUNT - 1, gx + 1); xx++) {
        for (const j of buckets[yy * CELL_COUNT + xx]!) {
          if (i === j || species[i] !== species[j]) continue
          const other = j * STATE_SIZE
          const dx = x - state[other]!, dy = y - state[other + 1]!
          const distanceSquared = dx * dx + dy * dy
          if (distanceSquared > NEIGHBOR_RADIUS_SQUARED || distanceSquared < 1e-9) continue
          neighbors++
          const influence = 0.065 / (distanceSquared + 0.002)
          separationX += dx * influence
          separationY += dy * influence
          sumX += state[other]!
          sumY += state[other + 1]!
          sumVx += state[other + 2]!
          sumVy += state[other + 3]!
        }
      }
    }
    if (neighbors > 0) {
      inputs[dest + 4] = limit(separationX / neighbors, -1.5, 1.5)
      inputs[dest + 5] = limit(separationY / neighbors, -1.5, 1.5)
      inputs[dest + 6] = limit((sumX / neighbors - x) * 6, -1, 1)
      inputs[dest + 7] = limit((sumY / neighbors - y) * 6, -1, 1)
      inputs[dest + 8] = limit((sumVx / neighbors - vx) * 2, -1, 1)
      inputs[dest + 9] = limit((sumVy / neighbors - vy) * 2, -1, 1)
    }
    if (pointer.mode !== 'off') {
      const dx = pointer.x - x, dy = pointer.y - y
      const distanceSquared = dx * dx + dy * dy
      const sign = pointer.mode === 'attract' ? 1 : -1
      const intensity = pointer.mode === 'attract'
        ? 1.3 : 0.22 / (distanceSquared + 0.018)
      inputs[dest + 10] = limit(sign * dx * intensity, -1.8, 1.8)
      inputs[dest + 11] = limit(sign * dy * intensity, -1.8, 1.8)
    }
    inputs[dest + 12] = x < 0.12 ? (0.12 - x) * 12 : x > 0.88 ? (0.88 - x) * 12 : 0
    inputs[dest + 13] = y < 0.12 ? (0.12 - y) * 12 : y > 0.88 ? (0.88 - y) * 12 : 0
  }
  return inputs
}

/** Copy the circuit's x/y/vx/vy output wires back to the next simulation state. */
export function projectState(
  fullValues: Float32Array, count: number, width: number,
  outputWires: readonly [number, number, number, number],
): Float32Array {
  if (fullValues.length !== count * width) throw new Error('batch output length mismatch')
  const projected = new Float32Array(count * STATE_SIZE)
  for (let i = 0; i < count; i++) {
    const at = i * width, dest = i * STATE_SIZE
    let x = fullValues[at + outputWires[0]]!, y = fullValues[at + outputWires[1]]!
    let vx = fullValues[at + outputWires[2]]!, vy = fullValues[at + outputWires[3]]!
    if (![x,y,vx,vy].every(Number.isFinite)) throw new Error('nonfinite circuit output')
    const speed = Math.hypot(vx,vy)
    if (speed > 0.85) {vx = Math.fround(vx * 0.85 / speed);vy = Math.fround(vy * 0.85 / speed)}
    if (x < 0.008 || x > 0.992) {x = limit(x, 0.008, 0.992);vx = Math.fround(-vx * 0.6)}
    if (y < 0.008 || y > 0.992) {y = limit(y, 0.008, 0.992);vy = Math.fround(-vy * 0.6)}
    projected.set([x, y, vx, vy], dest)
  }
  return projected
}

/** Separate CPU reference path; this never calls the WebGPU evaluator. */
export function referenceFrame(state: Float32Array, species: Uint8Array, pointer: PointerState) {
  const {circuit, outputWires} = buildSwarmCircuit()
  const inputs = buildSwarmInputs(state, species, pointer)
  const width = circuit.sourceSize + circuit.operations.length
  const values = new Float32Array(species.length * width)
  for (let i = 0; i < species.length; i++) {
    const local = circuit.evaluateReference(inputs.subarray(i * SOURCE_SIZE, (i + 1) * SOURCE_SIZE))
    values.set(local, i * width)
  }
  return {state: projectState(values, species.length, width, outputWires), values, inputs}
}

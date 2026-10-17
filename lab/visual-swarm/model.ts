import {Circuit} from '../../src/runtime/numeric-ui-core.ts'

/**
 * The flock-neighborhood reduction is a conventional, independently testable host stage.
 * Handwired evaluates the actual per-agent steering/integration arithmetic on CPU or WebGPU.
 * Do not advertise this as an entirely GPU-resident pairwise boids implementation.
 */
export const SOURCE_SIZE = 14
export const STATE_SIZE = 4
export const swarmConstants = {
  separation: 1.5, cohesion: 0.75, alignment: 0.6,
  pointer: 0.4, walls: 1.5, damping: 0.985,
  accelerationStep: 0.04, positionStep: 0.018,
} as const

export function buildSwarmCircuit() {
  const circuit = new Circuit(SOURCE_SIZE)
  const s = Array.from({length: SOURCE_SIZE}, (_, index) => circuit.source(index))
  const c = swarmConstants
  const ax = circuit.roundAffine([
    [s[4]!, c.separation], [s[6]!, c.cohesion], [s[8]!, c.alignment],
    [s[10]!, c.pointer], [s[12]!, c.walls],
  ])
  const ay = circuit.roundAffine([
    [s[5]!, c.separation], [s[7]!, c.cohesion], [s[9]!, c.alignment],
    [s[11]!, c.pointer], [s[13]!, c.walls],
  ])
  const vx = circuit.roundAffine([[s[2]!, c.damping], [ax, c.accelerationStep]])
  const vy = circuit.roundAffine([[s[3]!, c.damping], [ay, c.accelerationStep]])
  const x = circuit.roundAffine([[s[0]!, 1], [vx, c.positionStep]])
  const y = circuit.roundAffine([[s[1]!, 1], [vy, c.positionStep]])
  return {circuit, outputs: {x, y, vx, vy}, outputWires: [x, y, vx, vy] as const}
}

/** Independent FP32 arithmetic reference: deliberately never invokes Handwired. */
export function handwrittenStep(source: Float32Array): Float32Array {
  if (source.length !== SOURCE_SIZE) throw new Error('expected one 14-cell agent input')
  const f = Math.fround, c = swarmConstants
  const axis = (i: number) => {
    const coefficients = [c.separation, c.cohesion, c.alignment, c.pointer, c.walls]
    let sum = f(0)
    for (let k = 0; k < 5; k++) {
      sum = f(sum + f(source[4 + k * 2 + i]! * f(coefficients[k]!)))
    }
    return sum
  }
  const ax = axis(0), ay = axis(1)
  const vx = f(f(source[2]! * f(c.damping)) + f(ax * f(c.accelerationStep)))
  const vy = f(f(source[3]! * f(c.damping)) + f(ay * f(c.accelerationStep)))
  const x = f(f(source[0]!) + f(vx * f(c.positionStep)))
  const y = f(f(source[1]!) + f(vy * f(c.positionStep)))
  return Float32Array.of(x, y, vx, vy)
}

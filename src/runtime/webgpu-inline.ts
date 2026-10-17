import type {Circuit, Wire} from './numeric-ui-core.ts'

/**
 * Emit a pure WGSL function from an existing Handwired Circuit.
 *
 * Unlike the batch evaluator, this form has no fixed GPU storage bindings.
 * Its caller can gather sources from GPU-resident field buffers and consume
 * the outputs inside a larger stencil/physics compute pass without readbacks.
 *
 * Supports the same four native operation kinds as the general evaluator.
 */
export function emitInlineCircuitWgsl(
  circuit: Circuit, outputWires: readonly Wire[],
  functionName = 'executeHandwiredInline',
): string {
  const sourceSize = circuit.sourceSize
  const fullWidth = sourceSize + circuit.operations.length
  if (sourceSize < 1 || outputWires.length < 1 ||
    !/^[A-Za-z_][A-Za-z0-9_]*$/.test(functionName)) {
    throw new Error('invalid inline circuit function or empty circuit')
  }
  for (const wire of outputWires) {
    if (!Number.isInteger(wire) || wire < 0 || wire >= fullWidth) {
      throw new Error('invalid inline output wire')
    }
  }
  const bitPattern = (value: number) => {
    const view = new DataView(new ArrayBuffer(4))
    view.setFloat32(0, Math.fround(value), true)
    return 'bitcast<f32>(0x' + view.getUint32(0, true).toString(16).padStart(8, '0') + 'u)'
  }
  const lines = [
    'fn ' + functionName + '(sourceCells:array<f32,' + sourceSize + '>) -> array<f32,' + outputWires.length + '> {',
    '  var wireValues:array<f32,' + fullWidth + '>;',
    '  for (var i:u32=0u; i<' + sourceSize + 'u; i=i+1u) { wireValues[i]=sourceCells[i]; }',
  ]
  const emitAffine = (name: string, terms: readonly (readonly [Wire, number])[], bias: number) => {
    lines.push('  var ' + name + ':f32=' + bitPattern(bias) + ';')
    for (const [wire, coefficient] of terms) {
      lines.push('  ' + name + '=' + name + '+(' + bitPattern(coefficient) +
        '*wireValues[' + wire + 'u]);')
    }
  }
  for (const [index, op] of circuit.operations.entries()) {
    const value = 'v' + index
    emitAffine(value, op.leftTerms, op.leftBias)
    if (op.kind === 'reglu') {
      const gate = 'g' + index
      emitAffine(gate, op.gateTerms, op.gateBias)
      lines.push('  ' + value + '=' + value + '*max(0.0,' + gate + ');')
    } else if (op.kind === 'boolean') {
      lines.push('  ' + value + '=select(0.0,1.0,' + value + '>=0.5);')
    } else if (op.kind === 'index') {
      const gate = 'g' + index
      emitAffine(gate, op.gateTerms, op.gateBias)
      const read = op.indexRead!
      const candidates = read.table.candidates
      const below = read.below === 'zero' ? '0.0' : 'wireValues[' + candidates[0] + 'u]'
      const above = read.above === 'zero' ? '0.0' : 'wireValues[' + candidates[candidates.length - 1] + 'u]'
      lines.push('  if (isNan(' + gate + ')) {' + value + '=' + gate + ';}')
      lines.push('  else if (' + gate + '<0.0) {' + value + '=' + below + ';}')
      lines.push('  else if (' + gate + '>=' + bitPattern(candidates.length) +
        ') {' + value + '=' + above + ';}')
      lines.push('  else {')
      lines.push('    let selected=u32(floor(' + gate + '));')
      lines.push('    switch(selected) {')
      for (let c = 0; c < candidates.length; c++) {
        lines.push('      case ' + c + 'u: {' + value + '=wireValues[' + candidates[c] + 'u];}')
      }
      lines.push('      default: {' + value + '=' + above + ';}')
      lines.push('    }')
      lines.push('  }')
    }
    lines.push('  wireValues[' + (sourceSize + index) + 'u]=' + value + ';')
  }
  lines.push('  var outputs:array<f32,' + outputWires.length + '>;')
  for (const [index, wire] of outputWires.entries()) {
    lines.push('  outputs[' + index + 'u]=wireValues[' + wire + 'u];')
  }
  lines.push('  return outputs;')
  lines.push('}')
  return lines.join('\n') + '\n'
}

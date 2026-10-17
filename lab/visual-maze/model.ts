import {Circuit} from '../../src/runtime/numeric-ui-core.ts'

/** One parallel breadth-first-search expansion per grid cell.
 * Sources: visited, north/east/south/west frontier, passable.
 * The graph computes frontier OR, obstacle gating, visited exclusion, and visit update.
 * Neighbour gathering and final path backtracking are explicit host operations.
 */
export const MAZE_SOURCE_SIZE = 6
export function buildMazeCircuit(){
  const c = new Circuit(MAZE_SOURCE_SIZE)
  const visited=c.source(0), passable=c.source(5)
  const n=c.roundAffine([[c.source(1),1],[c.source(2),1],
    [c.source(3),1],[c.source(4),1]])
  // Inputs are binary, so clamp the 0..4 neighbour count to 0..1.
  const excess=c.reglu([[n,1]],-1,[],1)
  const hasFrontier=c.roundAffine([[n,1],[excess,-1]])
  const reachable=c.multiply(hasFrontier,passable)
  const unseen=c.roundAffine([[visited,-1]],1)
  const nextFrontier=c.multiply(reachable,unseen)
  const nextVisited=c.roundAffine([[visited,1],[nextFrontier,1]])
  return {circuit:c,outputs:{nextFrontier,nextVisited},outputWires:[nextVisited,nextFrontier] as const}
}

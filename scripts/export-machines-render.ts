import {mkdirSync,writeFileSync} from 'node:fs'
import {legPopulationRenderWgsl} from '../lab/frontiers/machines/render-shader.ts'
const path=process.argv[2]??'/kaggle/working'
mkdirSync(path,{recursive:true})
writeFileSync(path+'/machines-visual.wgsl',legPopulationRenderWgsl)
console.log('Exported real GPU-resident 3D articulated renderer vertex/fragment WGSL')

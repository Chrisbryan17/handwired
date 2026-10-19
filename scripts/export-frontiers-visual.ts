import {mkdirSync,writeFileSync} from 'node:fs'
import {cataclysmRenderWgsl} from '../lab/frontiers/cataclysm/render-shader.ts'
import {fractureVisualWgsl} from '../lab/frontiers/destruction/fracture-renderer.ts'
import {legPopulationRenderWgsl} from '../lab/frontiers/machines/render-shader.ts'
const out=process.argv[2]??'/kaggle/working'
mkdirSync(out,{recursive:true})
for(const [name,source] of Object.entries({
 'cataclysm-water':cataclysmRenderWgsl,
 'destruction-bridge':fractureVisualWgsl,
 'machines-robots':legPopulationRenderWgsl,
}))writeFileSync(out+'/'+name+'.wgsl',source)
console.log('Exported unchanged three actual Handwired WebGPU visual WGSL modules')

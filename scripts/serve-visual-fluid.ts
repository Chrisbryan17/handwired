import page from '../lab/visual-fluid/demo.html'

const port=Number(process.env['PORT']??'39085')
const server=Bun.serve({hostname:'127.0.0.1',port,routes:{'/':page},development:true})
console.log('Neural Fluid Visual Lab: '+server.url)

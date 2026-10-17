import page from '../lab/visual-cloth/demo.html'
const port=Number(process.env['PORT']??'39084')
const server=Bun.serve({hostname:'127.0.0.1',port,routes:{'/':page},development:true})
console.log('Neural Cloth Visual Lab: '+server.url)

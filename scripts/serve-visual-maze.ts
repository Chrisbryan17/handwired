import page from '../lab/visual-maze/demo.html'
const port=Number(process.env['PORT']??'39083')
const server=Bun.serve({hostname:'127.0.0.1',port,routes:{'/':page},development:true})
console.log('Neural Maze: '+server.url)

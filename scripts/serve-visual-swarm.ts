import page from '../lab/visual-swarm/demo.html'

const port = Number(process.env['PORT'] ?? '39081')
const server = Bun.serve({
  hostname: process.env['HOST'] ?? '127.0.0.1',
  port,
  routes: {'/': page},
  development: true,
})
console.log('Neural Swarm — Handwired Visual Lab: ' + server.url)

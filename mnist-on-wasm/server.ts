import { Hono } from 'hono'
import { serve } from '@hono/node-server'
import { serveStatic } from '@hono/node-server/serve-static'
import { readFileSync } from 'fs'

const app = new Hono()

// 静的ファイルの配信設定
// app.use('/*', serveStatic({ root: './index.html' }))
app.get('/', (c) => {
    return c.html(
        readFileSync('./assets/index.html', 'utf-8')
    )
})

// app.use('/pkg/*', serveStatic({ root: './pkg' }))
// app.use('/assets/*', serveStatic({ root: './assets' }))
app.use('/*', serveStatic({ root: './' }))

// サーバー起動
serve(app, (info) => {
  console.log(`Server is running on http://localhost:${info.port}`)
})

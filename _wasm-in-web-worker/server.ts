import { Hono } from 'hono'
import { serve } from '@hono/node-server'
import { serveStatic } from '@hono/node-server/serve-static'
import { readFileSync } from 'fs'

const app = new Hono()

// 静的ファイルの配信設定
// app.use('/*', serveStatic({ root: './index.html' }))
app.get('/', (c) => {
    return c.html(
        readFileSync('./index.html', 'utf-8')
    )
})

app.use('/*', serveStatic({ root: './' }))

// サーバー起動
serve(app, (info) => {
  console.log(`Server is running on http://localhost:${info.port}`)
})

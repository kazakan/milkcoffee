import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(__dirname, '../../www');
const port = Number(process.env.PORT || 4173);

const contentTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.wasm', 'application/wasm'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.svg', 'image/svg+xml'],
]);

// Proxy for external model downloads.  Browsers cannot fetch some remote
// URLs directly due to CORS restrictions (e.g. GitHub raw), so the app
// can route those requests through this server instead.
// Usage: GET /proxy-model?url=https://...
async function handleProxyModel(req, res) {
  const requestUrl = new URL(req.url, `http://127.0.0.1:${port}`);
  const target = requestUrl.searchParams.get('url');
  if (!target) {
    res.writeHead(400).end('Missing url parameter');
    return;
  }
  let parsed;
  try {
    parsed = new URL(target);
  } catch {
    res.writeHead(400).end('Invalid url parameter');
    return;
  }
  if (parsed.protocol !== 'https:') {
    res.writeHead(400).end('Only HTTPS URLs are proxied');
    return;
  }
  try {
    const upstream = await fetch(target);
    res.writeHead(upstream.status, {
      'Content-Type': upstream.headers.get('Content-Type') || 'application/octet-stream',
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'no-store',
    });
    const reader = upstream.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(value);
    }
    res.end();
  } catch (err) {
    if (!res.headersSent) res.writeHead(502).end('Upstream error: ' + err.message);
  }
}

const server = http.createServer(async (req, res) => {
  const requestUrl = new URL(req.url || '/', `http://127.0.0.1:${port}`);

  if (requestUrl.pathname === '/proxy-model') {
    await handleProxyModel(req, res);
    return;
  }

  const pathname = decodeURIComponent(requestUrl.pathname === '/' ? '/index.html' : requestUrl.pathname);
  const filePath = path.resolve(rootDir, `.${pathname}`);

  if (!filePath.startsWith(rootDir)) {
    res.writeHead(403).end('Forbidden');
    return;
  }

  try {
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) {
      res.writeHead(404).end('Not found');
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, {
      'Content-Type': contentTypes.get(ext) || 'application/octet-stream',
      'Cache-Control': 'no-store',
    });
    createReadStream(filePath).pipe(res);
  } catch {
    res.writeHead(404).end('Not found');
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Static server listening on http://127.0.0.1:${port}`);
});

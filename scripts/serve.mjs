#!/usr/bin/env node
// Serves demo/dist so the demo can be opened in a browser. `npm run demo` builds first.
//
//   node scripts/serve.mjs [--port 5173] [--open]

import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { readFile, stat } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const DIST = join(ROOT, 'demo', 'dist');

const argv = process.argv.slice(2);
const flag = (name) => {
  const index = argv.indexOf(name);
  return index === -1 ? undefined : argv[index + 1];
};
const PORT = Number(flag('--port') ?? process.env.PORT ?? 5173);
const OPEN = argv.includes('--open');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

if (!existsSync(join(DIST, 'index.html')) || !existsSync(join(DIST, 'app.js'))) {
  console.error(`demo/dist is not built. Run:  npm run build   (or just: npm run demo)`);
  process.exit(1);
}

const server = createServer(async (request, response) => {
  const path = decodeURIComponent((request.url ?? '/').split('?')[0]);
  const file = join(DIST, path === '/' ? 'index.html' : path);
  // resolve() collapses .. so a request cannot walk out of demo/dist
  if (!resolve(file).startsWith(DIST)) {
    response.writeHead(403).end('forbidden');
    return;
  }
  try {
    if (!(await stat(file)).isFile()) throw new Error('not a file');
    const body = await readFile(file);
    response.writeHead(200, {
      'content-type': TYPES[extname(file)] ?? 'application/octet-stream',
      'content-length': body.length,
      'cache-control': 'no-store',
    });
    response.end(body);
  } catch {
    response.writeHead(404).end('not found');
  }
});

server.on('error', (error) => {
  console.error(
    error.code === 'EADDRINUSE'
      ? `port ${PORT} is busy; try:  node scripts/serve.mjs --port 5174`
      : `server error: ${error.message}`,
  );
  process.exit(1);
});

server.listen(PORT, '127.0.0.1', async () => {
  const app = await stat(join(DIST, 'app.js'));
  const url = `http://127.0.0.1:${PORT}/`;
  console.log(`gpu-cron demo  ${url}`);
  console.log(`  app.js ${(app.size / 1024).toFixed(1)} KB (weights inlined)`);
  console.log('  needs WebGPU: Safari 26+, Chrome 113+, Edge 113+, Firefox 141+');
  console.log('  ctrl-c to stop');
  if (OPEN) {
    const opener = process.platform === 'darwin' ? 'open' : 'xdg-open';
    spawn(opener, [url], { stdio: 'ignore', detached: true }).on('error', () => {
      // headless box, or no handler registered for http; the URL above still works
    }).unref();
  }
});

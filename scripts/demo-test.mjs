#!/usr/bin/env node
// End-to-end smoke test: load the built demo in headless chromium, drive real prompts through
// the published parse(), and check that a valid cron and its fire times come back.
//
// The GPU conformance harness pins the arithmetic; this pins the thing a visitor actually
// does. It exercises the whole chain — inlined weights, WGSL shaders, the grammar automaton,
// the demo's own DOM.
//
//   node scripts/demo-test.mjs

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const DEMO_DIST = join(ROOT, 'demo', 'dist');
const CHROMIUM = process.env.CHROMIUM_BIN || 'chromium';

// An expression the constrained decoder cannot produce is the failure this is looking for.
const CASES = [
  { prompt: 'every 15 minutes', fields: 5 },
  { prompt: 'every weekday at 9am', fields: 5 },
  { prompt: 'first of the month at midnight', fields: 5 },
];

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
};

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

function serve(root) {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent((req.url ?? '/').split('?')[0]);
    const file = join(root, path === '/' ? 'index.html' : path);
    try {
      if (!(await stat(file)).isFile()) throw new Error('not a file');
      res.writeHead(200, { 'content-type': CONTENT_TYPES[extname(file)] ?? 'application/octet-stream' });
      res.end(await readFile(file));
    } catch {
      res.writeHead(404).end('not found');
    }
  });
  return new Promise((ready) => server.listen(0, '127.0.0.1', () => ready(server)));
}

class Cdp {
  constructor(ws) {
    this.ws = ws;
    this.next = 1;
    this.pending = new Map();
    ws.addEventListener('message', (event) => {
      const message = JSON.parse(event.data);
      const entry = this.pending.get(message.id);
      if (entry === undefined) return;
      this.pending.delete(message.id);
      entry(message);
    });
  }

  send(method, params = {}) {
    const id = this.next++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((done) => this.pending.set(id, done));
  }

  async evaluate(expression) {
    const reply = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (reply.error !== undefined) throw new Error(reply.error.message);
    const details = reply.result?.exceptionDetails;
    if (details !== undefined) throw new Error(details.exception?.description ?? details.text);
    return reply.result?.result?.value;
  }
}

async function connect(port, deadline) {
  while (Date.now() < deadline) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((target) => target.type === 'page' && target.webSocketDebuggerUrl);
      if (page !== undefined) {
        const ws = new WebSocket(page.webSocketDebuggerUrl);
        await new Promise((ready, fail) => {
          ws.addEventListener('open', ready, { once: true });
          ws.addEventListener('error', fail, { once: true });
        });
        return new Cdp(ws);
      }
    } catch {
      // devtools endpoint is not up yet
    }
    await sleep(150);
  }
  throw new Error('chromium never exposed a page target');
}

async function freePort() {
  const probe = await new Promise((ready) => {
    const server = createServer();
    server.listen(0, '127.0.0.1', () => ready(server));
  });
  const { port } = probe.address();
  await new Promise((done) => probe.close(done));
  return port;
}

async function run() {
  const server = await serve(DEMO_DIST);
  const base = `http://127.0.0.1:${server.address().port}`;
  const devtoolsPort = await freePort();
  const profile = await mkdtemp(join(tmpdir(), 'nl-cron-demo-'));
  const url = `${base}/index.html`;

  const child = spawn(
    CHROMIUM,
    [
      '--headless=new',
      '--enable-unsafe-webgpu',
      `--remote-debugging-port=${devtoolsPort}`,
      `--user-data-dir=${profile}`,
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-dev-shm-usage',
      '--disable-background-timer-throttling',
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] },
  );

  let cdp;
  try {
    cdp = await connect(devtoolsPort, Date.now() + 20_000);
    await cdp.send('Runtime.enable');
    await cdp.send('Page.enable');
    await cdp.send('Page.navigate', { url });
    await sleep(3000);

    const results = [];
    for (const testCase of CASES) {
      const outcome = await cdp.evaluate(`(async () => {
        const input = document.getElementById('input');
        // clearing the previous answer is what makes the poll below wait for THIS prompt
        document.getElementById('cron').textContent = '';
        document.getElementById('error-body').textContent = '';
        input.value = ${JSON.stringify(testCase.prompt)};
        document.getElementById('ask').dispatchEvent(new Event('submit', { cancelable: true }));
        const deadline = Date.now() + 240_000;
        while (Date.now() < deadline) {
          await new Promise((r) => setTimeout(r, 250));
          const cron = document.getElementById('cron').textContent.trim();
          if (cron !== '') return { cron, fires: document.getElementById('fires').children.length };
          const err = document.getElementById('error-body').textContent.trim();
          if (!document.getElementById('error').hidden && err !== '') return { error: err };
        }
        return { error: 'timed out waiting for a result' };
      })()`);
      results.push({ prompt: testCase.prompt, ...outcome });
    }

    let failed = 0;
    for (const [index, result] of results.entries()) {
      const expected = CASES[index]?.fields ?? 5;
      if (result.error !== undefined) {
        failed += 1;
        console.log(`  FAIL  ${result.prompt}\n        ${result.error}`);
        continue;
      }
      const fields = String(result.cron).trim().split(/\s+/);
      const ok = fields.length === expected && result.fires === 5;
      if (!ok) failed += 1;
      console.log(
        `  ${ok ? 'ok  ' : 'FAIL'}  ${result.prompt.padEnd(28)} -> ${result.cron}` +
          `   (${fields.length} fields, ${result.fires} fire times)`,
      );
    }

    // read the badge a visitor sees rather than a test-only hook on the page
    const backend = await cdp.evaluate(
      "document.body.innerText.match(/webgpu · [^\n]*|cpu · plain TS/)?.[0]?.trim() ?? 'unknown'",
    );
    console.log(`  backend: ${backend}`);
    console.log(failed === 0 ? 'PASS' : `FAIL (${failed}/${results.length})`);
    return failed === 0 ? 0 : 1;
  } finally {
    cdp?.ws.close();
    child.kill('SIGKILL');
    await new Promise((done) => server.close(done));
    // chromium writes its profile for a moment after the kill; a leftover tmpdir is not a failure
    await sleep(500);
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

process.exit(await run());

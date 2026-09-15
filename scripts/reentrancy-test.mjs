#!/usr/bin/env node
// Pins the two fixes the built bundles must carry: the shipped checkpoint, and parse() reentrancy.

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CHROMIUM = process.env.CHROMIUM_BIN || 'chromium';

const TUESDAY = 'every tuesday at 3 pm';
// A `*` in the day-of-week field means the checkpoint bundled here cannot express a weekday.
const EXPECTED_TUESDAY = '0 15 * * 2';
const PROMPTS = [TUESDAY, 'every 15 minutes', 'every weekday at 9am', 'every hour', 'first of the month at midnight'];

// The page only needs to exist on the origin so the bundle can be imported into it.
const PAGE = '<!doctype html><meta charset="utf-8"><title>reentrancy-test</title>';

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
};

const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

function serve(root) {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent((req.url ?? '/').split('?')[0]);
    if (path === '/') {
      res.writeHead(200, { 'content-type': CONTENT_TYPES['.html'] });
      res.end(PAGE);
      return;
    }
    const file = join(root, path);
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

// Sequential first so the concurrency below runs against a warm runtime, not a cold one.
const SCRIPT = `(async () => {
  const PROMPTS = ${JSON.stringify(PROMPTS)};
  const bundle = await import('/dist/index.js');
  if (!(await bundle.isAvailable())) return { error: 'isAvailable() resolved false: no usable WebGPU backend' };

  const sequential = [];
  for (const prompt of PROMPTS) sequential.push((await bundle.parse(prompt)).expression);

  const concurrent = [];
  let threw = null;
  try {
    const results = await Promise.all(PROMPTS.map((prompt) => bundle.parse(prompt)));
    for (const result of results) concurrent.push(result.expression);
  } catch (error) {
    threw = String(error?.message ?? error);
  }
  return { sequential, concurrent, threw };
})()`;

function withTimeout(promise, ms, note) {
  return Promise.race([
    promise,
    sleep(ms).then(() => {
      throw new Error(`timed out after ${ms}ms: ${note}`);
    }),
  ]);
}

function report(summary) {
  console.log(JSON.stringify(summary));
  return summary.weights === 'pass' && summary.reentrancy === 'pass' ? 0 : 1;
}

async function run() {
  const server = await serve(ROOT);
  const base = `http://127.0.0.1:${server.address().port}`;
  const devtoolsPort = await freePort();
  const profile = await mkdtemp(join(tmpdir(), 'gpu-cron-reentrancy-'));

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

  let stderr = '';
  child.stderr.on('data', (chunk) => {
    stderr = (stderr + chunk.toString()).slice(-4000);
  });

  let cdp;
  try {
    cdp = await connect(devtoolsPort, Date.now() + 20_000);
    await cdp.send('Runtime.enable');
    await cdp.send('Page.enable');
    await cdp.send('Page.navigate', { url: `${base}/` });
    await sleep(1000);

    const adapter = await withTimeout(
      cdp.evaluate("(async () => (navigator.gpu ? !!(await navigator.gpu.requestAdapter()) : false))()"),
      20_000,
      'probing for a WebGPU adapter',
    );
    if (adapter !== true) {
      console.error('environment: no WebGPU adapter in headless chromium (navigator.gpu + --enable-unsafe-webgpu)');
      console.error(`chromium stderr:\n${stderr}`);
      return report({ weights: 'skip', reentrancy: 'skip', error: 'no WebGPU adapter' });
    }

    const outcome = await withTimeout(cdp.evaluate(SCRIPT), TIMEOUT, 'running the checks in the page');
    if (outcome === undefined || outcome === null) throw new Error('the page returned no result');
    if (outcome.error !== undefined) {
      console.error(`environment: ${outcome.error}`);
      return report({ weights: 'skip', reentrancy: 'skip', error: outcome.error });
    }

    const tuesday = outcome.sequential[0];
    const weights = tuesday === EXPECTED_TUESDAY ? 'pass' : 'fail';
    const matching =
      outcome.threw === null &&
      outcome.concurrent.length === PROMPTS.length &&
      outcome.sequential.every((expression, index) => expression === outcome.concurrent[index]);
    const reentrancy = matching ? 'pass' : 'fail';

    for (const [index, prompt] of PROMPTS.entries()) {
      const concurrent = outcome.concurrent[index];
      console.log(
        `  ${prompt.padEnd(30)} sequential ${String(outcome.sequential[index]).padEnd(12)}` +
          ` concurrent ${concurrent === undefined ? '<none>' : concurrent}`,
      );
    }
    if (outcome.threw !== null) console.log(`  concurrent parse threw: ${outcome.threw}`);
    if (weights === 'fail') {
      console.log(`  weights: ${TUESDAY} gave ${JSON.stringify(tuesday)}, wanted ${JSON.stringify(EXPECTED_TUESDAY)}`);
    }

    return report({ weights, reentrancy, tuesday });
  } finally {
    cdp?.ws.close();
    child.kill('SIGKILL');
    await new Promise((done) => server.close(done));
    await sleep(500);
    await rm(profile, { recursive: true, force: true }).catch(() => {});
  }
}

const timeoutArg = process.argv.slice(2).find((arg) => arg.startsWith('--timeout='));
const TIMEOUT = timeoutArg === undefined ? 300_000 : Number(timeoutArg.slice('--timeout='.length));

process.exit(
  await run().catch((error) => {
    console.error(`FAIL: ${error?.stack ?? error}`);
    console.log(JSON.stringify({ weights: 'fail', reentrancy: 'fail', error: String(error?.message ?? error) }));
    return 1;
  }),
);

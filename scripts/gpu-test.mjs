#!/usr/bin/env node
// Headless-Chromium runner for test/gpu-conformance.html.
//
// Chromium will not hand out a WebGPU adapter under one fixed set of flags across machines, so
// this probes flag sets in order and uses the first that yields an adapter. If none do (CI box
// with no GPU and no software fallback) it prints SKIP and exits 0 -- a box without WebGPU must
// not fail the suite. Any other failure exits non-zero.
//
//   node scripts/gpu-test.mjs            # run the page's own window.makeLogits
//   node scripts/gpu-test.mjs --stub     # echo the reference logits back (proves the harness)

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const TEST_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'test');
const CHROMIUM = process.env.CHROMIUM_BIN || 'chromium';
const PAGE = 'gpu-conformance.html';
const NO_ADAPTER = 'no WebGPU adapter';

const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.wasm': 'application/wasm',
};

// The first three are the documented defaults. This box's chromium 149 returned a null adapter
// for all three and only produced one once Dawn's unsafe-adapter opt-in was added, so that is
// appended here rather than replacing the list.
const FLAG_SETS = [
  { name: 'default headless', flags: [] },
  { name: '--enable-unsafe-swiftshader', flags: ['--enable-unsafe-swiftshader'] },
  {
    name: '--use-angle=swiftshader --enable-unsafe-swiftshader --enable-features=Vulkan',
    flags: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--enable-features=Vulkan'],
  },
  { name: '--enable-unsafe-webgpu', flags: ['--enable-unsafe-webgpu'] },
  {
    name: '--enable-unsafe-webgpu --enable-unsafe-swiftshader',
    flags: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader'],
  },
  {
    name: '--enable-unsafe-webgpu --enable-unsafe-swiftshader --no-sandbox',
    flags: ['--enable-unsafe-webgpu', '--enable-unsafe-swiftshader', '--no-sandbox'],
  },
];

const STUB = `(async () => {
  const reference = await (await fetch('fixtures/tiny-reference.json')).json();
  await window.runConformance((_bytes, _manifest) => (ids) => {
    const key = ids.join(',');
    const hit = reference.cases.find((testCase) => testCase.ids.join(',') === key);
    if (!hit) throw new Error('stub has no reference case for ids ' + key);
    return Float32Array.from(hit.logits);
  });
})()`;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function parseArgs(argv) {
  const options = { stub: false, timeout: 60_000 };
  for (const arg of argv) {
    if (arg === '--stub') options.stub = true;
    else if (arg.startsWith('--timeout=')) options.timeout = Number(arg.slice('--timeout='.length));
    else if (arg === '--help' || arg === '-h') options.help = true;
    else {
      console.error(`unknown argument: ${arg}`);
      options.help = true;
    }
  }
  return options;
}

function freePort() {
  const probe = createServer();
  return new Promise((res) => {
    probe.listen(0, '127.0.0.1', () => {
      const { port } = probe.address();
      probe.close(() => res(port));
    });
  });
}

class Devtools {
  constructor(ws) {
    this.ws = ws;
    this.nextId = 1;
    this.pending = new Map();
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      const settle = this.pending.get(message.id);
      if (settle) {
        this.pending.delete(message.id);
        settle(message);
      }
    };
  }

  static async connect(url) {
    const ws = new WebSocket(url);
    await new Promise((res, rej) => {
      ws.onopen = res;
      ws.onerror = () => rej(new Error(`cannot connect to devtools at ${url}`));
    });
    return new Devtools(ws);
  }

  send(method, params) {
    const id = this.nextId;
    this.nextId += 1;
    return new Promise((res) => {
      this.pending.set(id, res);
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  async evaluate(expression) {
    const message = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    const details = message.result?.exceptionDetails;
    if (details) {
      throw new Error(details.exception?.description || details.text || 'evaluate threw');
    }
    return message.result?.result?.value;
  }

  close() {
    this.ws.close();
  }
}

async function waitForTarget(devtoolsPort, deadline) {
  while (Date.now() < deadline) {
    await sleep(200);
    try {
      const res = await fetch(`http://127.0.0.1:${devtoolsPort}/json/list`);
      const targets = await res.json();
      const page = targets.find((t) => t.type === 'page' && t.webSocketDebuggerUrl);
      if (page) return page;
    } catch {
      // devtools endpoint not up yet
    }
  }
  return null;
}

async function pollValue(cdp, expression, deadline, accept) {
  while (Date.now() < deadline) {
    let value;
    try {
      value = await cdp.evaluate(expression);
    } catch {
      value = undefined;
    }
    const hit = accept(value);
    if (hit !== undefined) return hit;
    await sleep(200);
  }
  return undefined;
}

// Returns {kind:'no-adapter'} | {kind:'result', result} | {kind:'timeout', note}
async function attempt(flagSet, options, deadline) {
  const origin = options.origin;
  const devtoolsPort = await freePort();
  const userDataDir = await mkdtemp(join(tmpdir(), 'gpu-test-'));
  const args = [
    '--headless=new',
    ...flagSet.flags,
    `--remote-debugging-port=${devtoolsPort}`,
    `--user-data-dir=${userDataDir}`,
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-dev-shm-usage',
    '--disable-background-timer-throttling',
    `${origin}/${PAGE}`,
  ];
  const child = spawn(CHROMIUM, args, { stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', (chunk) => {
    stderr = (stderr + chunk.toString()).slice(-4000);
  });

  let cdp = null;
  try {
    const target = await waitForTarget(devtoolsPort, deadline);
    if (!target) return { kind: 'timeout', note: 'chromium never exposed a devtools page target' };
    cdp = await Devtools.connect(target.webSocketDebuggerUrl);
    await cdp.send('Runtime.enable');

    const ready = await pollValue(cdp, 'window.harnessReady === true', deadline, (v) => (v === true ? true : undefined));
    if (ready === undefined) return { kind: 'timeout', note: 'harness never became ready' };

    const probe = await cdp.evaluate('window.probeAdapter()');
    if (!probe || probe.ok !== true) {
      const error = String(probe?.error ?? 'probeAdapter returned nothing');
      if (error.startsWith(NO_ADAPTER)) return { kind: 'no-adapter', error };
      return { kind: 'timeout', note: error, stderr };
    }

    const hasOwn = (await cdp.evaluate('typeof window.makeLogits')) === 'function';
    if (!options.stub && !hasOwn) {
      return {
        kind: 'timeout',
        note:
          'page exposes no window.makeLogits and --stub was not passed: implement the WGSL forward ' +
          'pass on the page, or run with --stub to exercise the harness itself',
      };
    }

    // Kick off the run without awaiting it: a hung forward pass must not hang the runner.
    void cdp.evaluate(options.stub ? STUB : 'window.runConformance(window.makeLogits)').catch(() => {});

    const result = await pollValue(
      cdp,
      'document.title',
      deadline,
      (value) => {
        if (typeof value !== 'string' || value === 'pending') return undefined;
        try {
          const parsed = JSON.parse(value);
          return typeof parsed.ok === 'boolean' ? parsed : undefined;
        } catch {
          return undefined;
        }
      },
    );
    if (result === undefined) {
      return { kind: 'timeout', note: 'harness never published a result', stderr };
    }
    return { kind: 'result', result, adapter: probe.adapter, stderr };
  } finally {
    cdp?.close();
    child.kill('SIGKILL');
    await rm(userDataDir, { recursive: true, force: true });
  }
}

const options = parseArgs(process.argv.slice(2));
if (options.help) {
  console.log('usage: node scripts/gpu-test.mjs [--stub] [--timeout=ms]');
  process.exit(0);
}
if (!Number.isFinite(options.timeout) || options.timeout <= 0) {
  console.error('--timeout must be a positive number of milliseconds');
  process.exit(2);
}

const server = createServer(async (req, res) => {
  const pathname = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const target = join(TEST_DIR, pathname === '/' ? PAGE : pathname);
  if (!target.startsWith(TEST_DIR)) {
    res.writeHead(403, { 'content-type': 'text/plain' });
    res.end('forbidden');
    return;
  }
  try {
    const body = await readFile(target);
    res.writeHead(200, { 'content-type': CONTENT_TYPES[extname(target)] || 'application/octet-stream' });
    res.end(body);
  } catch {
    res.writeHead(404, { 'content-type': 'text/plain' });
    res.end(`not found: ${pathname}`);
  }
});
// 127.0.0.1 rather than a LAN address: WebGPU needs a secure context.
await new Promise((res) => server.listen(0, '127.0.0.1', res));
const origin = `http://127.0.0.1:${server.address().port}`;

const deadline = Date.now() + options.timeout;
let adapterFound = false;
let exitCode = 1;
let lastNote = '';

for (const flagSet of FLAG_SETS) {
  if (Date.now() >= deadline) {
    lastNote = 'ran out of time probing flag sets';
    break;
  }
  const outcome = await attempt(flagSet, { ...options, origin }, deadline);
  if (outcome.kind === 'no-adapter') {
    console.log(`no adapter (${flagSet.name}): ${outcome.error}`);
    continue;
  }
  if (outcome.kind === 'timeout') {
    lastNote = outcome.note;
    if (outcome.stderr) lastNote += `\n--- chromium stderr ---\n${outcome.stderr}`;
    break;
  }
  adapterFound = true;
  console.log(`adapter ok via: --headless=new ${flagSet.flags.join(' ')}`.trim());
  console.log(JSON.stringify(outcome.result));
  exitCode = outcome.result.ok === true ? 0 : 1;
  break;
}

server.close();

if (!adapterFound && exitCode === 1) {
  if (lastNote === '') {
    console.log(`SKIP: no WebGPU adapter in headless chromium (${FLAG_SETS.length} flag sets tried)`);
    exitCode = 0;
  } else {
    console.error(`FAIL: ${lastNote}`);
  }
}

process.exit(exitCode);

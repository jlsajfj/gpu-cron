#!/usr/bin/env node
// Emits two bundles: dist/index.js (the published package — one file, weights inlined) and
// demo/dist/ (the Vercel site, which imports the package the same way a user would).

import { build, context } from 'esbuild';
import { watch as watchFs } from 'node:fs';
import { cp, mkdir, rm, stat, writeFile, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { brotliCompressSync } from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const SRC = path.join(ROOT, 'src');
const DIST = path.join(ROOT, 'dist');
const DEMO = path.join(ROOT, 'demo');
const DEMO_DIST = path.join(DEMO, 'dist');
const GENERATED = path.join(SRC, 'weights.generated.ts');
const MODEL =
  process.env.NL_CRON_MODEL !== undefined
    ? path.resolve(process.env.NL_CRON_MODEL).replace(/\.(bin|json)$/, '')
    : path.join(ROOT, 'weights', 'model');

const argv = new Set(process.argv.slice(2));
const WATCH = argv.has('--watch');
const WEIGHTS_ONLY = argv.has('--weights-only');
const PRODUCTION = argv.has('--minify') || process.env.NODE_ENV === 'production';

const bytes = (n) =>
  n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(2)} MB`;
const gzip = (buffer) => brotliCompressSync(buffer).length;
const line = (name, size, extra = '') => console.log(`  ${name.padEnd(22)} ${bytes(size).padStart(10)}${extra}`);
const packed = (buffer) => `   ${bytes(gzip(buffer))} brotli`;

async function sizeOf(file) {
  return (await stat(file)).size;
}

// The weights are inlined so the package has no fetch, no assets and no runtime to download.
// A missing model is not a build error: the module reports it absent and parse() says so.
async function writeWeights() {
  const binPath = `${MODEL}.bin`;
  const manifestPath = `${MODEL}.json`;
  const name = path.basename(binPath);
  if (!existsSync(binPath) || !existsSync(manifestPath)) {
    const placeholder = [
      `import type { ModelManifest } from './forward.js';`,
      `export const MODEL_BASE64 = '';`,
      `export const MODEL_BYTES = 0;`,
      `export const MODEL_PRESENT = false;`,
      `export const MODEL_NAME = ${JSON.stringify(name)};`,
      `export const MANIFEST: ModelManifest = { d_model: 0, n_layer: 0, n_head: 0, d_ff: 0, max_len: 0, vocab: 0, tensors: [] };`,
      '',
    ].join('\n');
    await writeFile(GENERATED, placeholder, 'utf8');
    console.log(`  ${'weights'.padEnd(22)} ${'no model'.padStart(10)}   (set NL_CRON_MODEL or add ${name})`);
    return;
  }
  const buffer = await readFile(binPath);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const source = [
    `import type { ModelManifest } from './forward.js';`,
    `export const MODEL_BASE64 = ${JSON.stringify(buffer.toString('base64'))};`,
    `export const MODEL_BYTES = ${buffer.length};`,
    `export const MODEL_PRESENT = true;`,
    `export const MODEL_NAME = ${JSON.stringify(name)};`,
    `export const MANIFEST: ModelManifest = ${JSON.stringify(manifest)};`,
    '',
  ].join('\n');
  await writeFile(GENERATED, source, 'utf8');
  line(`weights (${name})`, buffer.length, packed(buffer));
}

function bundleOptions(entry, outfile) {
  return {
    entryPoints: [entry],
    outfile,
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    sourcemap: PRODUCTION ? false : 'linked',
    minify: PRODUCTION,
    legalComments: 'none',
    logLevel: 'warning',
  };
}

async function bundle(entry, outfile) {
  const result = await build(bundleOptions(entry, outfile));
  if (result.errors.length > 0) throw new Error(`${result.errors.length} error(s) in ${path.basename(outfile)}`);
  return readFile(outfile);
}

async function copyDemo() {
  await rm(DEMO_DIST, { recursive: true, force: true });
  await mkdir(DEMO_DIST, { recursive: true });
  await cp(path.join(DEMO, 'index.html'), path.join(DEMO_DIST, 'index.html'));
  await cp(path.join(DEMO, 'styles.css'), path.join(DEMO_DIST, 'styles.css'));
  const publicDir = path.join(DEMO, 'public');
  if (existsSync(publicDir)) await cp(publicDir, DEMO_DIST, { recursive: true });
  for (const rel of ['index.html', 'styles.css']) {
    line(`demo/${rel}`, await sizeOf(path.join(DEMO_DIST, rel)));
  }
}

async function buildOnce() {
  await mkdir(DIST, { recursive: true });
  const lib = await bundle(path.join(SRC, 'index.ts'), path.join(DIST, 'index.js'));
  line('dist/index.js', lib.length, packed(lib));
  await cp(path.join(ROOT, 'scripts', 'index.d.ts'), path.join(DIST, 'index.d.ts'));
  line('dist/index.d.ts', await sizeOf(path.join(DIST, 'index.d.ts')));
  await copyDemo();
  const app = await bundle(path.join(DEMO, 'app.ts'), path.join(DEMO_DIST, 'app.js'));
  line('demo/dist/app.js', app.length, packed(app));
  const conf = await bundle(path.join(ROOT, 'test', 'gpu-entry.ts'), path.join(ROOT, 'test', 'gpu-bundle.js'));
  line('test/gpu-bundle.js', conf.length, packed(conf));
}

async function watchAll() {
  await mkdir(DIST, { recursive: true });
  await mkdir(DEMO_DIST, { recursive: true });
  const lib = await context(bundleOptions(path.join(SRC, 'index.ts'), path.join(DIST, 'index.js')));
  await lib.watch();
  const app = await context(bundleOptions(path.join(DEMO, 'app.ts'), path.join(DEMO_DIST, 'app.js')));
  await app.watch();
  const conf = await context(
    bundleOptions(path.join(ROOT, 'test', 'gpu-entry.ts'), path.join(ROOT, 'test', 'gpu-bundle.js')),
  );
  await conf.watch();
  let pending = null;
  watchFs(DEMO, { recursive: true }, (_event, file) => {
    if (file !== null && file.startsWith('dist')) return;
    clearTimeout(pending);
    pending = setTimeout(() => {
      copyDemo().catch((err) => console.error(`  demo copy failed: ${err.message}`));
    }, 50);
  });
}

try {
  if (WEIGHTS_ONLY) {
    await writeWeights();
    console.log('done (weights only)');
  } else {
    if (!WATCH) await rm(DIST, { recursive: true, force: true });
    await mkdir(DIST, { recursive: true });
    await writeWeights();
    if (WATCH) {
      console.log('building -> dist/ + demo/dist/');
      await copyDemo();
      await watchAll();
      console.log('watching for changes (ctrl-c to stop)');
    } else {
      console.log(`building${PRODUCTION ? ' (production)' : ''} -> dist/ + demo/dist/`);
      await buildOnce();
      console.log('done');
    }
  }
} catch (err) {
  console.error(`build failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}

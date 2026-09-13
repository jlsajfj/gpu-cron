#!/usr/bin/env node
// Bundles the demo into web/dist: app.js, the inlined int8 model, ORT's wasm assets, index.html.

import { context } from 'esbuild';
import { watch as watchFs } from 'node:fs';
import { cp, mkdir, readdir, rm, stat, writeFile, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const DIST = path.join(ROOT, 'dist');
const PUBLIC = path.join(ROOT, 'public');
const SRC = path.join(ROOT, 'src');
const WEIGHTS = 'weights.generated.js';
// export/export_js.py writes a .bin plus a .json manifest; the page reads both from one
// inlined module, so there is no fetch and no inference runtime to download.
const MODEL =
  process.env.WEB_MODEL !== undefined
    ? path.resolve(process.env.WEB_MODEL).replace(/\.(bin|json)$/, '')
    : path.join(ROOT, 'weights', 'model');

const argv = new Set(process.argv.slice(2));
const WATCH = argv.has('--watch');
const PRODUCTION = argv.has('--minify') || process.env.NODE_ENV === 'production';

const bytes = (n) => (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(1)} KB` : `${(n / (1024 * 1024)).toFixed(2)} MB`);
const line = (name, size) => console.log(`  ${name.padEnd(34)} ${bytes(size).padStart(10)}`);

async function sizeOf(file) {
  return (await stat(file)).size;
}

function options() {
  return {
    entryPoints: [path.join(SRC, 'app.ts')],
    outfile: path.join(DIST, 'app.js'),
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

// The model is inlined so the page needs no fetch of its own; a missing model still builds —
// the generated module reports it absent and the app shows the no-model panel.
async function writeWeights() {
  const binPath = `${MODEL}.bin`;
  const manifestPath = `${MODEL}.json`;
  const name = path.basename(binPath);
  const out = path.join(DIST, WEIGHTS);
  if (!existsSync(binPath) || !existsSync(manifestPath)) {
    const placeholder = [
      `export const MODEL_BASE64 = '';`,
      `export const MODEL_BYTES = 0;`,
      `export const MODEL_PRESENT = false;`,
      `export const MODEL_NAME = ${JSON.stringify(name)};`,
      `export const MANIFEST = { d_model: 0, n_layer: 0, n_head: 0, d_ff: 0, max_len: 0, vocab: 0, tensors: [] };`,
      '',
    ].join('\n');
    await writeFile(out, placeholder, 'utf8');
    console.log(`  ${WEIGHTS.padEnd(34)} ${'no model'.padStart(10)}   (set WEB_MODEL or add ${name})`);
    return;
  }
  const buffer = await readFile(binPath);
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const source = [
    `export const MODEL_BASE64 = ${JSON.stringify(buffer.toString('base64'))};`,
    `export const MODEL_BYTES = ${buffer.length};`,
    `export const MODEL_PRESENT = true;`,
    `export const MODEL_NAME = ${JSON.stringify(name)};`,
    `export const MANIFEST = ${JSON.stringify(manifest)};`,
    '',
  ].join('\n');
  await writeFile(out, source, 'utf8');
  line(WEIGHTS, await sizeOf(out));
}

async function copyStatic() {
  await cp(path.join(ROOT, 'index.html'), path.join(DIST, 'index.html'));
  line('index.html', await sizeOf(path.join(DIST, 'index.html')));
  await cp(path.join(SRC, 'styles.css'), path.join(DIST, 'styles.css'));
  line('styles.css', await sizeOf(path.join(DIST, 'styles.css')));
  if (!existsSync(PUBLIC)) return;
  await cp(PUBLIC, DIST, { recursive: true });
  for (const rel of await readdir(PUBLIC, { recursive: true })) {
    const file = path.join(PUBLIC, rel);
    if (!(await stat(file)).isFile()) continue;
    line(rel, await sizeOf(path.join(DIST, rel)));
  }
}

async function buildOnce() {
  const ctx = await context(options());
  try {
    const result = await ctx.rebuild();
    if (result.errors.length > 0) throw new Error(`${result.errors.length} build error(s)`);
    line('app.js', await sizeOf(path.join(DIST, 'app.js')));
  } finally {
    await ctx.dispose();
  }
}

async function watchAll() {
  const ctx = await context({
    ...options(),
    plugins: [
      {
        name: 'report',
        setup(build) {
          build.onEnd(async (result) => {
            if (result.errors.length > 0) {
              console.error('  app.js FAILED');
              return;
            }
            line('app.js', await sizeOf(path.join(DIST, 'app.js')));
          });
        },
      },
    ],
  });
  await ctx.watch();

  const watched = [SRC, path.join(ROOT, 'index.html'), PUBLIC, path.dirname(MODEL)];
  for (const dir of watched) {
    if (!existsSync(dir)) continue;
    let pending = null;
    watchFs(dir, { recursive: true }, () => {
      clearTimeout(pending);
      pending = setTimeout(() => {
        copyStatic()
          .then(() => (dir === path.dirname(MODEL) ? writeWeights() : undefined))
          .catch((err) => console.error(`  copy failed: ${err.message}`));
      }, 50);
    });
  }
}

try {
  if (!WATCH) await rm(DIST, { recursive: true, force: true });
  await mkdir(DIST, { recursive: true });
  console.log(`${PRODUCTION ? 'building (production)' : 'building'} -> dist/`);

  await copyStatic();
  await writeWeights();

  if (WATCH) {
    await watchAll();
    console.log('watching for changes (ctrl-c to stop)');
  } else {
    await buildOnce();
    console.log('done');
  }
} catch (err) {
  console.error(`build failed: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
}

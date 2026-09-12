#!/usr/bin/env node
// Bundles the demo into web/dist: app.js (automaton + decoder + UI), the base64 int8 model,
// the ONNX Runtime wasm assets, index.html and anything in web/public.

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
const ORT_DIST = path.join(ROOT, 'node_modules', 'onnxruntime-web', 'dist');
const ORT_ASSETS = ['ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.jsep.mjs'];
const WEIGHTS = 'weights.generated.js';
const MODEL = process.env.WEB_MODEL ? path.resolve(process.env.WEB_MODEL) : path.join(ROOT, 'weights', 'model.int8.onnx');

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

// The model is inlined so the page needs no fetch of its own, and a missing model still
// builds: the generated module reports it absent and the app shows the no-onnx panel.
async function writeWeights() {
  const name = path.relative(ROOT, MODEL);
  const out = path.join(DIST, WEIGHTS);
  if (!existsSync(MODEL)) {
    const placeholder = [
      `export const MODEL_BASE64 = '';`,
      `export const MODEL_BYTES = 0;`,
      `export const MODEL_PRESENT = false;`,
      `export const MODEL_NAME = ${JSON.stringify(name)};`,
      '',
    ].join('\n');
    await writeFile(out, placeholder, 'utf8');
    console.log(`  ${WEIGHTS.padEnd(34)} ${'no model'.padStart(10)}   (set WEB_MODEL or add ${name})`);
    return;
  }
  const buffer = await readFile(MODEL);
  const source = [
    `export const MODEL_BASE64 = ${JSON.stringify(buffer.toString('base64'))};`,
    `export const MODEL_BYTES = ${buffer.length};`,
    `export const MODEL_PRESENT = true;`,
    `export const MODEL_NAME = ${JSON.stringify(name)};`,
    '',
  ].join('\n');
  await writeFile(out, source, 'utf8');
  line(WEIGHTS, await sizeOf(out));
}

async function copyOrtAssets() {
  const dir = path.join(DIST, 'ort');
  await mkdir(dir, { recursive: true });
  for (const asset of ORT_ASSETS) {
    const from = path.join(ORT_DIST, asset);
    if (!existsSync(from)) {
      console.warn(`  missing onnxruntime-web asset: ${asset}`);
      continue;
    }
    const to = path.join(dir, asset);
    if (existsSync(to) && (await sizeOf(to)) === (await sizeOf(from))) continue;
    await cp(from, to);
    line(path.join('ort', asset), await sizeOf(to));
  }
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
  await copyOrtAssets();
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

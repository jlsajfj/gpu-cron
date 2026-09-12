// Stage 1 of the dataset: cron expression -> canonical English, via cronstrue.
// PAIR_TARGET is the size of the *finished* dataset (canonicals x phrasings), not the
// number of distinct expressions; paraphrasing happens in stage 2.
// Emits one row per *distinct* cron string; paraphrasing happens in stage 2.

import { createWriteStream } from 'node:fs';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import cronstrue from 'cronstrue';
import {
  BUCKETS,
  isViable,
  assertFieldCount,
  mulberry32,
  fireTimes,
} from './cron-distribution.mjs';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const OUT = path.join(ROOT, 'data', 'out');

const TARGET = Number(process.env.PAIR_TARGET ?? 250000);
const SEED = Number(process.env.SEED ?? 20260912);
const FIRE_SAMPLES = 8;

export function canonicalize(cron) {
  return cronstrue.toString(cron, {
    use24HourTimeFormat: false,
    dayOfWeekStartIndexZero: true,
    throwExceptionOnParseError: true,
    verbose: false,
  });
}

const rng = mulberry32(SEED);
const rows = [];
let attempts = 0;

// Quotas are on *pairs*, not expressions. Sampling N distinct expressions would let the
// four high-cardinality buckets crowd out "every 5 minutes" and "daily at 9am" — the
// shapes people actually write. Each bucket instead fills its own share of the pair
// budget, spending extra phrasings on the buckets that have few distinct expressions.
const BASE_PHRASINGS = 8;
const MAX_PHRASINGS = 40;
const TOTAL_WEIGHT = BUCKETS.reduce((sum, b) => sum + b.weight, 0);

function collectBucket(bucket, wantDistinct) {
  const seen = new Set();
  const out = [];
  const cap = wantDistinct * 40 + 2000;
  let tries = 0;
  while (out.length < wantDistinct && tries < cap) {
    tries += 1;
    attempts += 1;
    const cron = bucket.sample(rng);
    if (seen.has(cron)) continue;
    seen.add(cron);
    try {
      assertFieldCount(cron);
      // cron-parser is the only gate that catches "parses fine, never fires" (Feb 30).
      if (!isViable(cron)) continue;
      const english = canonicalize(cron);
      const fires = fireTimes(cron, FIRE_SAMPLES);
      if (fires === null) continue;
      out.push({ cron, english, bucket: bucket.name, fires });
    } catch {
      continue;
    }
  }
  return out;
}

for (const bucket of BUCKETS) {
  const quotaPairs = Math.round((TARGET * bucket.weight) / TOTAL_WEIGHT);
  const collected = collectBucket(bucket, Math.ceil(quotaPairs / BASE_PHRASINGS));
  const phrasings = Math.min(
    MAX_PHRASINGS,
    Math.max(BASE_PHRASINGS, Math.ceil(quotaPairs / Math.max(1, collected.length))),
  );
  for (const row of collected) rows.push({ ...row, phrasings });
}

const projectedPairs = rows.reduce((sum, r) => sum + r.phrasings, 0);
if (projectedPairs < TARGET * 0.5) {
  throw new Error(`only projected ${projectedPairs}/${TARGET} pairs after ${attempts} attempts`);
}

// Seeded shuffle so bucket order can't leak into a positional train/val/test split.
for (let i = rows.length - 1; i > 0; i -= 1) {
  const j = Math.floor(rng() * (i + 1));
  [rows[i], rows[j]] = [rows[j], rows[i]];
}

const n = rows.length;
const nTest = Math.floor(n * 0.04);
const nVal = Math.floor(n * 0.03);
const splitOf = (i) => (i < nTest ? 'test' : i < nTest + nVal ? 'val' : 'train');

await mkdir(OUT, { recursive: true });
const out = createWriteStream(path.join(OUT, 'canonical.jsonl'));
let i = 0;
for (const row of rows) {
  out.write(`${JSON.stringify({ ...row, split: splitOf(i) })}\n`);
  i += 1;
}
await new Promise((resolve) => out.end(resolve));

const stats = new Map();
for (const row of rows) {
  const s = stats.get(row.bucket) ?? { n: 0, pairs: 0, phrasings: row.phrasings };
  s.n += 1;
  s.pairs += row.phrasings;
  stats.set(row.bucket, s);
}
const bySplit = { train: 0, val: 0, test: 0 };
rows.forEach((_, idx) => (bySplit[splitOf(idx)] += 1));
const totalPairs = rows.reduce((sum, r) => sum + r.phrasings, 0);

console.log(`wrote ${n} canonical rows to data/out/canonical.jsonl`);
console.log(`  attempts=${attempts}  (${((1 - n / attempts) * 100).toFixed(1)}% of draws were dupes/unviable)`);
console.log(`  splits: ${JSON.stringify(bySplit)}`);
console.log(`  projected pairs: ${totalPairs}`);
console.log('  buckets:');
for (const [name, s] of [...stats.entries()].sort((a, b) => b[1].pairs - a[1].pairs)) {
  console.log(
    `    ${name.padEnd(22)} ${String(s.n).padStart(6)} exprs  x${String(s.phrasings).padStart(2)}  ` +
      `${String(s.pairs).padStart(7)} pairs  ${((s.pairs / totalPairs) * 100).toFixed(1)}%`,
  );
}

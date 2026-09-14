// Stage 2b: deterministic coverage of number surface forms.
//
// The LLM paraphraser is uneven about how it spells numbers -- it produced 1559 "fifteenth"
// and zero "second" -- so the model learned surface form, not value+role. This pass walks the
// grid (value x spelling x cron field) directly so every form of every number appears in every
// role it can legally play. Seeded; regenerating gives byte-identical output.

import { writeFile, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mulberry32, isViable } from './cron-distribution.mjs';

const ROOT = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const OUT = path.join(ROOT, 'data', 'out');

const ORD_WORD = [null, 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh',
  'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth',
  'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'twenty-first',
  'twenty-second', 'twenty-third', 'twenty-fourth', 'twenty-fifth', 'twenty-sixth',
  'twenty-seventh', 'twenty-eighth', 'twenty-ninth', 'thirtieth', 'thirty-first'];

const CARD_WORD = [null, 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
  'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
  'eighteen', 'nineteen', 'twenty', 'twenty-one', 'twenty-two', 'twenty-three', 'twenty-four',
  'twenty-five', 'twenty-six', 'twenty-seven', 'twenty-eight', 'twenty-nine', 'thirty',
  'thirty-one'];

function ordNum(n) {
  const rem100 = n % 100;
  if (rem100 >= 11 && rem100 <= 13) return `${n}th`;
  return `${n}${({ 1: 'st', 2: 'nd', 3: 'rd' })[n % 10] ?? 'th'}`;
}

// Every way a day-of-month value can be written, so none of them is the model's only route
// to the day-of-month field.
function domForms(d) {
  return [`the ${ordNum(d)}`, `the ${ORD_WORD[d]}`, `day ${d}`, `the ${ordNum(d)} day`,
    `the ${ORD_WORD[d]} day`, `day ${CARD_WORD[d]}`];
}

function clock(h, m) {
  const suffix = h < 12 ? 'am' : 'pm';
  const h12 = h % 12 === 0 ? 12 : h % 12;
  const pad = String(m).padStart(2, '0');
  if (h === 0 && m === 0) return ['midnight'];
  if (h === 12 && m === 0) return ['noon'];
  if (m === 0) return [`${h12}${suffix}`, `${h12} ${suffix}`, `${String(h).padStart(2, '0')}:00`];
  return [`${h12}:${pad}${suffix}`, `${String(h).padStart(2, '0')}:${pad}`];
}

export function buildRows() {
  const rows = [];
  const push = (cron, bucket, phrasings) => {
    if (!isViable(cron)) throw new Error(`augment produced non-viable cron: ${cron}`);
    const uniq = [...new Set(phrasings.filter(Boolean))];
    if (uniq.length >= 2) rows.push({ cron, bucket, phrasings: uniq });
  };

// 1. Day-of-month with NO time stated. The model currently echoes the day digit into the hour
// ("15th of the month" -> "0 15 15 * *"); these pin an unstated time to midnight.
for (let d = 1; d <= 31; d += 1) {
  push(`0 0 ${d} * *`, 'aug-dom-notime', [
    ...domForms(d).map((f) => `on ${f} of the month`),
    ...domForms(d).map((f) => `${f} of the month`),
    ...domForms(d).map((f) => `monthly on ${f}`),
    `every month on ${ordNum(d)}`,
    `run on ${ORD_WORD[d]} of each month`,
  ]);
}

// 2. Day-of-month WITH a time, so the hour field has its own separate evidence.
const rng = mulberry32(0x5eed);
for (let d = 1; d <= 31; d += 1) {
  for (let rep = 0; rep < 16; rep += 1) {
    const h = Math.floor(rng() * 24);
    const m = [0, 0, 15, 30, 45][Math.floor(rng() * 5)];
    const times = clock(h, m);
    push(`${m} ${h} ${d} * *`, 'aug-dom-time', [
      ...domForms(d).flatMap((f) => times.map((t) => `at ${t} on ${f} of the month`)),
      ...times.map((t) => `${ORD_WORD[d]} of the month at ${t}`),
      ...times.map((t) => `${ordNum(d)} of the month at ${t}`),
    ]);
  }
}

// 3. Steps. "every second hour" and "every other hour" are the step sense of the same words;
// both were absent, and "every other hour" was landing the step in the minute field.
const STEP_UNITS = [
  { unit: 'hour', plural: 'hours', cron: (n) => `0 */${n} * * *`, max: 12 },
  { unit: 'minute', plural: 'minutes', cron: (n) => `*/${n} * * * *`, max: 30 },
  { unit: 'day', plural: 'days', cron: (n) => `0 0 */${n} * *`, max: 15 },
];
// Each step value is generated at several minute/hour offsets: the step words need enough
// mass to beat the plain "every N <unit>" prior, and offsets give that without repetition.
const STEP_OFFSETS = [0, 5, 10, 15, 20, 30, 40, 45];
for (const { unit, plural, cron, max } of STEP_UNITS) {
  for (let n = 2; n <= max; n += 1) {
    for (const off of STEP_OFFSETS) {
      if (off !== 0) {
        const shifted = cron(n).replace(/^\S+/, String(off));
        if (unit === 'minute') continue;
        const at = unit === 'hour'
          ? [`at ${off} past`, `${off} minutes past the hour`]
          : [`at ${String(off).padStart(2, '0')}:00`];
        push(shifted, `aug-step-${unit}`, [
          ...at.map((a) => `every ${n} ${plural} ${a}`),
          ...at.map((a) => `every ${ORD_WORD[n]} ${unit} ${a}`),
          ...at.map((a) => `every ${ordNum(n)} ${unit} ${a}`),
          ...(n === 2 ? at.map((a) => `every other ${unit} ${a}`) : []),
        ]);
      }
    }
    const phrasings = [
      `every ${n} ${plural}`, `every ${CARD_WORD[n]} ${plural}`, `every ${ordNum(n)} ${unit}`,
      `every ${ORD_WORD[n]} ${unit}`, `once every ${n} ${plural}`,
      `once every ${CARD_WORD[n]} ${plural}`, `at every ${ordNum(n)} ${unit}`,
      `every ${n} ${plural} around the clock`, `each ${ordNum(n)} ${unit}`,
      `each ${ORD_WORD[n]} ${unit}`, `run every ${ORD_WORD[n]} ${unit}`,
      `repeat every ${n} ${plural}`, `${n}-${unit}ly`, `at ${n}-${unit} intervals`,
      `at ${CARD_WORD[n]}-${unit} intervals`,
    ];
    if (n === 2) phrasings.push(`every other ${unit}`, `every second ${unit}`, `on alternate ${plural}`);
    push(cron(n), `aug-step-${unit}`, phrasings);
  }
}

// 4. Hour-only, so a bare number after "at" stays in the hour field.
for (let h = 0; h <= 23; h += 1) {
  push(`0 ${h} * * *`, 'aug-hour', [
    ...clock(h, 0).map((t) => `every day at ${t}`),
    ...clock(h, 0).map((t) => `daily at ${t}`),
    ...clock(h, 0).map((t) => `at ${t} every day`),
    `at ${h}:00 each day`,
  ]);
}

  return rows;
}

async function main() {
// Split assignment follows canonical.jsonl wherever an expression already appears there, so an
// augmented phrasing can never move a test expression into train.
const canonicalPath = path.join(OUT, 'canonical.jsonl');
const splitOf = new Map();
if (existsSync(canonicalPath)) {
  for (const line of (await readFile(canonicalPath, 'utf8')).split('\n')) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    if (row.split) splitOf.set(row.cron, row.split);
  }
}

// Expressions canonical.jsonl has never seen are split by a hash of the expression: stable
// across runs, and independent of the order this file generates them in.
function hashSplit(cron) {
  let h = 2166136261;
  for (let i = 0; i < cron.length; i += 1) {
    h ^= cron.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const r = (h >>> 0) / 4294967296;
  if (r < 0.06) return 'test';
  if (r < 0.12) return 'val';
  return 'train';
}

const rows = buildRows();
const out = rows.map((r) => ({
  cron: r.cron,
  english: r.phrasings[0],
  bucket: r.bucket,
  split: splitOf.get(r.cron) ?? hashSplit(r.cron),
  phrasings: r.phrasings,
}));

const counts = {};
for (const r of out) counts[r.split] = (counts[r.split] ?? 0) + r.phrasings.length;
await writeFile(path.join(OUT, 'paraphrase.ckpt.augment.jsonl'),
  out.map((r) => JSON.stringify({ rows: [r] })).join('\n') + '\n');
console.log(`augment: ${out.length} expressions, ${out.reduce((a, r) => a + r.phrasings.length, 0)} phrasings`);
console.log(`  by split: ${JSON.stringify(counts)}`);
const byBucket = {};
for (const r of out) byBucket[r.bucket] = (byBucket[r.bucket] ?? 0) + 1;
console.log(`  by bucket: ${JSON.stringify(byBucket)}`);
}

const isMain = process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop());
if (isMain) await main();

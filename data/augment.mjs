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
    `${ordNum(d)} of the month`, `${ORD_WORD[d]} of the month`,
    `${ordNum(d)} of every month`, `${ORD_WORD[d]} of every month`,
    ...domForms(d).map((f) => `monthly on ${f}`),
    `every month on ${ordNum(d)}`,
    `run on ${ORD_WORD[d]} of each month`,
  ]);
}

// 1b. The spelled ordinal lives almost entirely in bucket 1, which is capped at 31
// expressions (one per day) and was outweighed by the time buckets. Crossing the day with a
// month scope multiplies the expressions without inventing a time the phrasing never stated.
const MONTH_SCOPES = [
  { m: '1', words: ['of January', 'in January', 'each January'] },
  { m: '6', words: ['of June', 'in June'] },
  { m: '12', words: ['of December', 'in December'] },
  { m: '1,4,7,10', words: ['of each quarter month', 'in January, April, July and October'] },
  { m: '1-6', words: ['in the first half of the year'] },
];
for (let d = 1; d <= 31; d += 1) {
  for (const { m, words } of MONTH_SCOPES) {
    // Short months: day 31 of June never fires, and the dataset must not teach a date that
    // cannot happen. isViable is the same cron-parser check build-canonical.mjs uses.
    if (!isViable(`0 0 ${d} ${m} *`)) continue;
    push(`0 0 ${d} ${m} *`, 'aug-dom-notime', words.flatMap((w) => [
      `on the ${ordNum(d)} ${w}`, `on the ${ORD_WORD[d]} ${w}`,
      `the ${ordNum(d)} ${w}`, `the ${ORD_WORD[d]} ${w}`,
      `${ordNum(d)} ${w}`, `${ORD_WORD[d]} ${w}`,
      `day ${d} ${w}`,
    ]));
  }
}

// 1c. Ordinals against the minute field: "the 12th minute" is minute 12, not hour 12 and not
// a step. Without this the only ordinals the model sees are day-of-month ones.
for (let m = 0; m <= 59; m += 1) {
  push(`${m} * * * *`, 'aug-minute-ordinal', [
    `at the ${ordNum(m)} minute`, `on the ${ordNum(m)} minute`,
    `the ${ordNum(m)} minute of every hour`, `at the ${ORD_WORD[m] ?? ordNum(m)} minute`,
    `the ${ORD_WORD[m] ?? ordNum(m)} minute of each hour`,
    `${m} minutes past every hour`, `at ${m} past the hour`,
    `every hour at ${ordNum(m)} minute past`,
  ]);
}

// 1d. Months as a sole value. February, April, May, August and October only ever appeared
// inside a list or a range, so "8am on august 3rd" had no shape to copy.
const MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
  'September', 'October', 'November', 'December'];
const MONTH_DAYS = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
const mrng = mulberry32(0x3007);
for (let mo = 1; mo <= 12; mo += 1) {
  const name = MONTH_NAMES[mo - 1];
  for (const d of [1, 5, 10, 15, 20, 25, MONTH_DAYS[mo - 1]]) {
    for (let rep = 0; rep < 6; rep += 1) {
      const h = Math.floor(mrng() * 24);
      const h12 = h % 12 === 0 ? 12 : h % 12;
      const ap = h < 12 ? 'am' : 'pm';
      push(`0 ${h} ${d} ${mo} *`, 'aug-month', [
        `at ${h12}${ap} on ${name} ${ordNum(d)}`, `${name} ${ordNum(d)} at ${h12}${ap}`,
        `on the ${ordNum(d)} of ${name} at ${h12}${ap}`,
        `every ${name} ${ordNum(d)} at ${h12}${ap}`,
        `at ${h12}${ap} on the ${ORD_WORD[d]} of ${name}`,
      ]);
    }
    push(`0 9 ${d} ${mo} *`, 'aug-month', [
      `9am on ${name} ${ordNum(d)}`, `${name} ${ordNum(d)}`, `every ${name} ${ordNum(d)} at 9am`,
      `on ${name} ${ordNum(d)} at 9 in the morning`,
    ]);
  }
  push(`0 8 * ${mo} *`, 'aug-month', [
    `every day in ${name} at 8am`, `daily during ${name} at 8am`,
    `8am every day of ${name}`, `each day in ${name} at 8am`,
  ]);
}

// 1e. Day-of-month lists written with ordinals, and the thin day abbreviations
// ("tues" had 160 rows, "weds" 13).
const DOM_PAIRS = [[1, 15], [1, 16], [5, 20], [10, 25], [1, 10, 20], [7, 14, 21, 28],
  [2, 16], [3, 18], [6, 21], [8, 23], [12, 26], [4, 14, 24], [1, 8, 15, 22], [9, 19, 29],
  [11, 21], [13, 27], [1, 20], [15, 30], [2, 12, 22], [5, 10, 15, 20, 25]];
for (const days of DOM_PAIRS) {
  const nums = days.map(ordNum);
  const wordsOrd = days.map((d) => ORD_WORD[d]);
  const j = (xs) => xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];
  push(`0 0 ${days.join(',')} * *`, 'aug-dom-list', [
    `on the ${j(nums)} of the month`, `on the ${j(wordsOrd)} of the month`,
    `the ${j(nums)} of each month`, `the ${j(wordsOrd)} of each month`,
    `monthly on the ${j(nums)}`, `on day ${j(days.map(String))} of the month`,
  ]);
}
const DOW_ABBR = [
  { n: 1, forms: ['mon', 'mon.'] }, { n: 2, forms: ['tue', 'tues', 'tue.'] },
  { n: 3, forms: ['wed', 'weds', 'wed.'] }, { n: 4, forms: ['thu', 'thur', 'thurs'] },
  { n: 5, forms: ['fri', 'fri.'] }, { n: 6, forms: ['sat'] }, { n: 0, forms: ['sun'] },
];
for (const { n, forms } of DOW_ABBR) {
  for (const hh of [6, 7, 8, 9, 12, 14, 17, 18, 20, 22]) {
    push(`0 ${hh} * * ${n}`, 'aug-dow-abbr', forms.flatMap((f) => [
      `at ${hh} on ${f}`, `every ${f} at ${hh}:00`, `${f} at ${hh}:00`,
    ]));
  }
  push(`0 9 * * ${n}`, 'aug-dow-abbr', forms.flatMap((f) => [
    `at 9 on ${f}`, `at 9am on ${f}`, `every ${f} at 9am`, `${f} at 9am`, `9am each ${f}`,
  ]));
  for (const { n: n2, forms: f2 } of DOW_ABBR) {
    if (n2 <= n) continue;
    push(`0 9 * * ${[n, n2].sort((a, b) => a - b).join(',')}`, 'aug-dow-abbr', [
      `every ${forms[0]} and ${f2[0]} at 9`, `at 9am on ${forms[0]} and ${f2[0]}`,
      `${forms[0]} and ${f2[0]} at 9am`,
    ]);
  }
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

// 3c. Step words crossed with day restrictions. The step phrasings were 3% of the
// augmentation and lost to the time buckets; "every second minute" had two rows in 136k.
const DAY_SCOPES = [
  { dow: '1-5', words: ['on weekdays', 'during the week', 'Monday through Friday'] },
  { dow: '0,6', words: ['on weekends', 'at the weekend'] },
  { dow: '1', words: ['on Mondays'] },
  { dow: '3', words: ['on Wednesdays'] },
  { dow: '5', words: ['on Fridays'] },
];
for (const { unit, plural, cron, max } of STEP_UNITS) {
  for (let n = 2; n <= max; n += 1) {
    for (const { dow, words } of DAY_SCOPES) {
      const base = cron(n).split(' ');
      base[4] = dow;
      push(base.join(' '), `aug-step-${unit}`, words.flatMap((w) => [
        `every ${n} ${plural} ${w}`, `every ${ORD_WORD[n]} ${unit} ${w}`,
        `every ${ordNum(n)} ${unit} ${w}`, `every ${CARD_WORD[n]} ${plural} ${w}`,
        ...(n === 2 ? [`every other ${unit} ${w}`] : []),
      ]));
    }
  }
}

// 3b. Times. The sampler only ever emitted minutes from {0,15,30,45}: 53 of the 60 values
// never appeared in training, so the model snapped arbitrary minutes to the nearest tidy one
// ("3:21pm" -> minute 15). Every minute value is covered here, in every clock style, plus the
// fraction words the digit-only phrasings can never teach.
const trng = mulberry32(0x7113);
for (let m = 0; m <= 59; m += 1) {
  for (let rep = 0; rep < 8; rep += 1) {
    const h = Math.floor(trng() * 24);
    const h12 = h % 12 === 0 ? 12 : h % 12;
    const ap = h < 12 ? 'am' : 'pm';
    const mm = String(m).padStart(2, '0');
    const forms = [`${h12}:${mm}${ap}`, `${h12}:${mm} ${ap}`, `${String(h).padStart(2, '0')}:${mm}`];
    const phrasings = [
      ...forms.map((f) => `every day at ${f}`),
      ...forms.map((f) => `daily at ${f}`),
      ...forms.map((f) => `at ${f} every day`),
      ...forms.map((f) => `run it at ${f} each day`),
    ];
    // A bare 12-hour clock with no am/pm is ambiguous ("quarter past 3" is 03:15 or 15:15).
    // Only the AM reading is generated, so one phrasing never carries two labels.
    if (h < 12) {
      if (m === 15) phrasings.push(`daily at quarter past ${h12}`, `every day at a quarter past ${h12}`);
      if (m === 30) phrasings.push(`daily at half past ${h12}`, `every day at half past ${h12}`);
      if (m === 45) phrasings.push(`daily at quarter to ${h12 + 1}`, `every day at a quarter to ${h12 + 1}`);
      if (m === 0) phrasings.push(`every day at ${h12} o'clock`, `daily at ${h12} o'clock`);
    }
    push(`${m} ${h} * * *`, 'aug-time', phrasings);
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

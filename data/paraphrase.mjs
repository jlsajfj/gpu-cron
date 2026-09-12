// Stage 2 of the dataset: canonical English -> many surface phrasings, via an LLM.
// Resumable: finished batches are appended to a checkpoint keyed by index.

import { createWriteStream, existsSync, readFileSync } from 'node:fs';
import { mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const OUT = path.join(ROOT, 'data', 'out');

// gpt-4.1-mini over gpt-5-mini: the reasoning model burns 2-3k hidden tokens per call (35s vs 6s).
const MODEL = process.env.PARAPHRASE_MODEL ?? 'gpt-4.1-mini';
const MAX_PHRASINGS_PER_CALL = 160;
const BATCH = Number(process.env.PARAPHRASE_BATCH ?? 8);
const CONCURRENCY = Number(process.env.PARAPHRASE_CONCURRENCY ?? 32);
const LIMIT = process.env.PARAPHRASE_LIMIT ? Number(process.env.PARAPHRASE_LIMIT) : Infinity;

const SYSTEM = `You expand schedule descriptions into the varied ways real people phrase them.

You receive canonical schedule strings (the kind a cron library emits) and return
natural-language phrasings a person would actually type into a "run this on a schedule" box.

Hard rules:
- Every phrasing must describe EXACTLY the same schedule: identical times of day, identical days.
- Never add or remove a day, a time, or an interval.
- Never mention timezones, UTC, seconds, years, or relative dates ("tomorrow", "next week").
- 12-hour clock must carry am/pm. Use "noon" for 12:00 PM and "midnight" for 12:00 AM.
- Mon-Fri is "weekdays". Saturday+Sunday is "weekends". A single day may be named ("on Tuesdays").
- Vary length from about 3 to 16 words, and vary register: terse fragments, "remind me to"
  requests, questions, and plain statements should all appear across the set.
- Vary clock style: some 12-hour with am/pm, some 24-hour where it reads naturally.
- Spread the phrasings over different openings; do not start most of them the same way.
- No markdown, no bullet characters, no trailing periods needed.
- Never repeat a phrasing within a row, and never reuse a phrasing across rows.

Reply with JSON only, shaped exactly as:
{"rows":[{"i":<index>,"phrasings":["...","..."]}]}
where "i" echoes the index you were given.`;

function buildUser(batch, count) {
  const payload = JSON.stringify(batch.map((row, i) => ({ i, canonical: row.english })));
  return `Canonical schedule strings (${batch.length} of them). Produce exactly ${count} distinct phrasings for each, in the given order.\n\n${payload}`;
}

async function callOpenAI(messages, attempt = 0) {
  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: MODEL,
      messages,
      response_format: { type: 'json_object' },
      max_completion_tokens: 8192,
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    if ((res.status === 429 || res.status >= 500) && attempt < 5) {
      await new Promise((r) => setTimeout(r, 1500 * 2 ** attempt));
      return callOpenAI(messages, attempt + 1);
    }
    throw new Error(`openai ${res.status}: ${body.slice(0, 200)}`);
  }
  const json = await res.json();
  const choice = json.choices?.[0];
  if (choice?.finish_reason === 'length') throw new Error('truncated response');
  return JSON.parse(choice.message.content);
}

const BANNED = /\b(utc|gmt|timezone|time zone|seconds?|tomorrow|yesterday|next week|next month)\b/i;

const hasScheduleWord =
  /\b(every|each|daily|weekly|monthly|weekday|weekend|hour|minute|day|days|month|months|noon|midnight|morning|afternoon|evening|night|mon|tue|wed|thu|fri|sat|sun)\b/i;

function cleanAll(phrasings, count) {
  const out = [];
  for (const raw of phrasings ?? []) {
    if (typeof raw !== 'string') continue;
    const s = raw.replace(/\s+/g, ' ').replace(/^[-*•]\s*/, '').trim();
    if (s.length < 3 || s.length > 160) continue;
    if (BANNED.test(s)) continue;
    if (!/[a-z]/i.test(s)) continue;
    if (!/[0-9]/.test(s) && !hasScheduleWord.test(s)) continue;
    out.push(s);
  }
  return [...new Set(out)].slice(0, count);
}

const canonicalPath = path.join(OUT, 'canonical.jsonl');
if (!existsSync(canonicalPath)) throw new Error('run build-canonical.mjs first');

const canonical = (await readFile(canonicalPath, 'utf8'))
  .split('\n')
  .filter(Boolean)
  .map((l) => JSON.parse(l));

const byCount = new Map();
for (const row of canonical) {
  const key = row.phrasings ?? 8;
  if (!byCount.has(key)) byCount.set(key, []);
  byCount.get(key).push(row);
}

const jobs = [];
for (const [count, group] of [...byCount.entries()].sort((a, b) => a[0] - b[0])) {
  const size = Math.max(1, Math.min(BATCH, Math.floor(MAX_PHRASINGS_PER_CALL / count)));
  for (let i = 0; i < group.length; i += size) {
    jobs.push({ count, rows: group.slice(i, i + size), index: jobs.length });
  }
}

const ckptPath = path.join(OUT, 'paraphrase.ckpt.jsonl');
const done = new Set();
if (existsSync(ckptPath)) {
  for (const line of readFileSync(ckptPath, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    try {
      done.add(JSON.parse(line).i);
    } catch {
      // a torn final line from a killed run; that batch is simply redone
    }
  }
}

const todo = jobs.filter((j) => !done.has(j.index)).slice(0, LIMIT === Infinity ? undefined : LIMIT);

console.log(
  `paraphrase: ${canonical.length} rows -> ${jobs.length} batches, ${todo.length} to do ` +
    `(model=${MODEL}, batch=${BATCH}, concurrency=${CONCURRENCY})`,
);

await mkdir(OUT, { recursive: true });
const ckpt = createWriteStream(ckptPath, { flags: 'a' });

let processed = 0;
let failures = 0;
let cursor = 0;
let emitted = 0;
const started = Date.now();

async function worker() {
  while (cursor < todo.length) {
    const job = todo[cursor];
    cursor += 1;
    try {
      const parsed = await callOpenAI([
        { role: 'system', content: SYSTEM },
        { role: 'user', content: buildUser(job.rows, job.count) },
      ]);
      const byIndex = new Map((parsed.rows ?? []).map((r) => [r.i, cleanAll(r.phrasings, job.count)]));
      const rows = job.rows.map((row, i) => ({
        cron: row.cron,
        english: row.english,
        bucket: row.bucket,
        split: row.split,
        phrasings: byIndex.get(i) ?? [],
      }));
      emitted += rows.reduce((sum, r) => sum + r.phrasings.length, 0);
      ckpt.write(`${JSON.stringify({ i: job.index, rows })}\n`);
      processed += 1;
    } catch (err) {
      failures += 1;
      console.error(`  batch ${job.index} failed: ${err.message}`);
    }
    const total = processed + failures;
    if (total % 20 === 0 && total > 0) {
      const rate = total / ((Date.now() - started) / 1000);
      console.log(
        `  ${total}/${todo.length} batches  pairs=${emitted}  fail=${failures}  ` +
          `eta ${Math.round((todo.length - total) / rate / 60)}m`,
      );
    }
  }
}

await Promise.all(Array.from({ length: CONCURRENCY }, worker));
await new Promise((resolve) => ckpt.end(resolve));

console.log(`done: ${processed} batches ok, ${failures} failed, ${emitted} raw pairs`);

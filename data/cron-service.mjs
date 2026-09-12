// Cron semantics over stdin/stdout, one JSON request per line, one JSON response per line.
// Everything that needs to know what a cron expression *means* goes through cron-parser
// here, so the training/eval Python never grows a second, subtly different implementation.
//
//   {"op":"validate","crons":["*/5 * * * *", ...]}
//   {"op":"fires","cron":"0 9 * * *","n":5,"from":"2026-01-01T00:00:00Z"}
//   {"op":"semantic","pairs":[["0 9 * * *","0 9 * * *"],...],"n":5}
//
// All enumeration is pinned to UTC and a fixed start date so results are reproducible.

import { createInterface } from 'node:readline';
import { CronExpressionParser } from 'cron-parser';

const TZ = 'UTC';
const DEFAULT_FROM = '2026-01-01T00:00:00Z';

export function fireTimes(cron, n, from = DEFAULT_FROM) {
  const out = [];
  try {
    const it = CronExpressionParser.parse(cron, { currentDate: new Date(from), tz: TZ });
    for (let i = 0; i < n; i += 1) out.push(it.next().toDate().toISOString());
  } catch {
    return null;
  }
  return out;
}

function validate(cron) {
  const times = fireTimes(cron, 3);
  return { cron, parses: times !== null, fires: times !== null && times.length === 3 };
}

const handlers = {
  validate: (req) => ({ results: (req.crons ?? []).map(validate) }),
  fires: (req) => ({ times: fireTimes(req.cron, req.n ?? 5, req.from) }),
  semantic: (req) => {
    const n = req.n ?? 5;
    const cache = new Map();
    const timesOf = (cron) => {
      if (!cache.has(cron)) cache.set(cron, fireTimes(cron, n, req.from));
      return cache.get(cron);
    };
    return {
      equal: (req.pairs ?? []).map(([a, b]) => {
        const ta = timesOf(a);
        const tb = timesOf(b);
        if (ta === null || tb === null) return false;
        return ta.length === tb.length && ta.every((t, i) => t === tb[i]);
      }),
    };
  },
};

// Importable as a module (eval/tests) and runnable as a line server (Python subprocess).
export function handle(request) {
  const handler = handlers[request?.op];
  if (handler === undefined) return { error: `unknown op: ${request?.op}` };
  try {
    return handler(request);
  } catch (err) {
    return { error: err instanceof Error ? err.message : String(err) };
  }
}

const isMain = process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop());
if (isMain) {
  const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
  rl.on('line', (line) => {
    if (!line.trim()) return;
    let response;
    try {
      response = handle(JSON.parse(line));
    } catch (err) {
      response = { error: `bad request: ${err.message}` };
    }
    process.stdout.write(`${JSON.stringify(response)}\n`);
  });
}

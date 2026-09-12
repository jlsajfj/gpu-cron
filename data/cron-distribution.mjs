// Samples cron expressions from a weighted, hand-tuned distribution of the shapes
// people actually write, then validates each one by enumerating fire times.

import { CronExpressionParser } from 'cron-parser';

export const FIELDS = ['minute', 'hour', 'dom', 'month', 'dow'];

const DOW_NAMES = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)];
const int = (rng, lo, hi) => lo + Math.floor(rng() * (hi - lo + 1));

const MINUTES = [0, 5, 10, 15, 20, 30, 45];
const TIDY_MINUTES = [0, 15, 30, 45];
const HOURS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23];

// `m` and `h` are only used by the buckets that need a time-of-day; most buckets
// roll their own so that "daily at HH:MM" stays denser than uniform over 24*60.
function time(rng) {
  return { m: pick(rng, TIDY_MINUTES), h: pick(rng, HOURS) };
}

// Every bucket writes exactly one restricted DOM/DOW field or neither. Restricting
// both flips cron semantics from AND to OR, which no phrasing in the dataset could
// disambiguate; see the README's ambiguity policies.
export const BUCKETS = [
  {
    name: 'every-minute',
    weight: 2,
    sample: () => '* * * * *',
  },
  {
    name: 'every-n-minutes',
    weight: 9,
    sample: (rng) => `*/${pick(rng, [2, 5, 10, 15, 20, 30])} * * * *`,
  },
  {
    name: 'offset-n-minutes',
    weight: 3,
    sample: (rng) => `${pick(rng, [1, 5, 10, 20, 25, 40])}/${pick(rng, [3, 6, 12, 15, 20])} * * * *`,
  },
  {
    name: 'every-hour',
    weight: 3,
    sample: () => '0 * * * *',
  },
  {
    name: 'every-n-hours',
    weight: 7,
    sample: (rng) => `${pick(rng, MINUTES)} */${pick(rng, [2, 3, 4, 6, 8, 12])} * * *`,
  },
  {
    name: 'daily-at',
    weight: 13,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} * * *`;
    },
  },
  {
    name: 'weekdays-at',
    weight: 11,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} * * 1-5`;
    },
  },
  {
    name: 'weekend-at',
    weight: 4,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} * * 0,6`;
    },
  },
  {
    name: 'weekly-on-day',
    weight: 8,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} * * ${int(rng, 0, 6)}`;
    },
  },
  {
    name: 'every-n-days',
    weight: 4,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} */${pick(rng, [2, 3, 4, 5, 7, 10, 14])} * *`;
    },
  },
  {
    name: 'monthly-on-day',
    weight: 6,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} ${pick(rng, [1, 2, 3, 5, 10, 14, 15, 20, 25, 28])} * *`;
    },
  },
  {
    name: 'monthly-on-dom-range',
    weight: 3,
    sample: (rng) => {
      const { m, h } = time(rng);
      const lo = int(rng, 1, 4);
      return `${m} ${h} ${lo}-${int(rng, lo + 3, 28)} * *`;
    },
  },
  {
    name: 'monthly-on-dom-list',
    weight: 4,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} ${pick(rng, ['1,15', '1,15,28', '5,20', '2,16'])} * *`;
    },
  },
  {
    name: 'yearly',
    weight: 3,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} ${pick(rng, [1, 15, 28])} ${pick(rng, [1, 3, 6, 7, 9, 11, 12])} *`;
    },
  },
  {
    name: 'hour-range',
    weight: 5,
    sample: (rng) => {
      const lo = int(rng, 0, 16);
      return `${pick(rng, TIDY_MINUTES)} ${lo}-${int(rng, lo + 2, 23)} * * *`;
    },
  },
  {
    name: 'hour-range-step',
    weight: 6,
    sample: (rng) => {
      const lo = int(rng, 0, 15);
      const step = pick(rng, [2, 3, 4, 6]);
      const tail = rng() < 0.5 ? '*' : pick(rng, ['1-5', '1-6', '1,3,5']);
      return `${pick(rng, [0, 15, 30])} ${lo}-${int(rng, lo + 3, 23)}/${step} * * ${tail}`;
    },
  },
  {
    name: 'hour-list',
    weight: 4,
    sample: (rng) =>
      `${pick(rng, TIDY_MINUTES)} ${pick(rng, ['9,17', '0,12', '8,12,18', '6,18', '0,6,12,18', '10,14,16'])} * * *`,
  },
  {
    name: 'minute-list',
    weight: 3,
    sample: (rng) => `${pick(rng, ['0,15,30,45', '0,30', '5,35', '20,40'])} ${pick(rng, HOURS)} * * *`,
  },
  {
    name: 'dow-range',
    weight: 5,
    sample: (rng) => {
      const lo = int(rng, 0, 4);
      return `${pick(rng, TIDY_MINUTES)} ${pick(rng, HOURS)} * * ${lo}-${int(rng, lo + 1, 6)}`;
    },
  },
  {
    name: 'dow-list',
    weight: 5,
    sample: (rng) => {
      const { m, h } = time(rng);
      return `${m} ${h} * * ${pick(rng, ['1,3,5', '2,4', '0,2,4', '1,2,3,4,5', '0,6', '3,6'])}`;
    },
  },
  {
    name: 'dow-step',
    weight: 3,
    sample: (rng) => `${pick(rng, TIDY_MINUTES)} ${pick(rng, HOURS)} * * */${pick(rng, [2, 3])}`,
  },
  {
    name: 'month-range',
    weight: 3,
    sample: (rng) => {
      const lo = int(rng, 1, 9);
      return `${pick(rng, TIDY_MINUTES)} ${pick(rng, HOURS)} ${pick(rng, [1, 15])} ${lo}-${int(rng, lo + 1, 12)} *`;
    },
  },
  {
    name: 'month-list',
    weight: 3,
    sample: (rng) => `${pick(rng, TIDY_MINUTES)} ${pick(rng, HOURS)} ${pick(rng, [1, 15, 28])} ${pick(rng, ['1,4,7,10', '3,6,9,12', '1,7', '11,12'])} *`,
  },
  {
    name: 'business-hours',
    weight: 5,
    sample: (rng) => `${pick(rng, [0, 15, 30])} 9-17 * * 1-5`,
  },
];

const TOTAL_WEIGHT = BUCKETS.reduce((sum, b) => sum + b.weight, 0);

export function sampleCron(rng) {
  let roll = rng() * TOTAL_WEIGHT;
  for (const bucket of BUCKETS) {
    roll -= bucket.weight;
    if (roll <= 0) return { cron: bucket.sample(rng), bucket: bucket.name };
  }
  const last = BUCKETS[BUCKETS.length - 1];
  return { cron: last.sample(rng), bucket: last.name };
}

const REQUIRED_FIRES = 3;

export function countFires(cron, { max = 1000 } = {}) {
  let n = 0;
  try {
    const it = CronExpressionParser.parse(cron, { currentDate: new Date() });
    while (n < max) {
      it.next();
      n += 1;
    }
  } catch {
    return n;
  }
  return n;
}

// cron-parser rejects impossible dates ("0 0 30 2 *") at parse time, and refuses to
// enumerate an expression that never fires. Every sample goes through here.
export function isViable(cron) {
  return countFires(cron, { max: REQUIRED_FIRES }) >= REQUIRED_FIRES;
}

export function fireTimes(cron, count, fromDate = new Date('2026-01-01T00:00:00Z')) {
  const out = [];
  try {
    const it = CronExpressionParser.parse(cron, { currentDate: fromDate });
    for (let i = 0; i < count; i += 1) out.push(it.next().toDate().toISOString());
  } catch {
    return null;
  }
  return out;
}

export function assertFieldCount(cron) {
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) throw new Error(`expected 5 fields, got ${parts.length}: ${cron}`);
  const [minute, hour, dom, , dow] = parts;
  const domRestricted = dom !== '*';
  const dowRestricted = dow !== '*';
  if (domRestricted && dowRestricted) {
    throw new Error(`both DOM and DOW restricted (OR-semantics trap): ${cron}`);
  }
  return { minute, hour, dom, dow };
}

export { DOW_NAMES };

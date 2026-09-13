export interface ParsedCron {
  minute: Set<number>;
  hour: Set<number>;
  dom: Set<number>;
  month: Set<number>;
  dow: Set<number>;
  domWildcard: boolean;
  dowWildcard: boolean;
}

function parseField(text: string, min: number, max: number): Set<number> | null {
  const out = new Set<number>();
  for (const term of text.split(',')) {
    let step = 1;
    let range = term;
    const slash = term.indexOf('/');
    if (slash >= 0) {
      range = term.slice(0, slash);
      step = Number(term.slice(slash + 1));
      if (!Number.isInteger(step) || step < 1) return null;
    }
    let lo: number;
    let hi: number;
    if (range === '*') {
      lo = min;
      hi = max;
    } else {
      const dash = range.indexOf('-');
      if (dash >= 0) {
        lo = Number(range.slice(0, dash));
        hi = Number(range.slice(dash + 1));
      } else {
        lo = Number(range);
        hi = slash >= 0 ? max : lo;
      }
    }
    if (!Number.isInteger(lo) || !Number.isInteger(hi)) return null;
    if (lo < min || hi > max || lo > hi) return null;
    for (let v = lo; v <= hi; v += step) out.add(v);
  }
  return out.size > 0 ? out : null;
}

export function parseCronExpression(cron: string): ParsedCron | null {
  const parts = cron.trim().split(/\s+/);
  if (parts.length !== 5) return null;
  const [minute, hour, domText, month, dowText] = parts as [string, string, string, string, string];
  const minuteValues = parseField(minute, 0, 59);
  const hourValues = parseField(hour, 0, 23);
  const dom = parseField(domText, 1, 31);
  const monthValues = parseField(month, 1, 12);
  const dow = parseField(dowText, 0, 6);
  if (!minuteValues || !hourValues || !dom || !monthValues || !dow) return null;
  return {
    minute: minuteValues,
    hour: hourValues,
    dom,
    month: monthValues,
    dow,
    domWildcard: domText === '*',
    dowWildcard: dowText === '*',
  };
}

// Cron's day rule: with both DOM and DOW restricted either may match (OR); a bare star
// makes the other field the only veto.
function dayMatches(parsed: ParsedCron, date: Date): boolean {
  const domMatch = parsed.dom.has(date.getDate());
  const dowMatch = parsed.dow.has(date.getDay());
  if (parsed.domWildcard && parsed.dowWildcard) return true;
  if (parsed.domWildcard) return dowMatch;
  if (parsed.dowWildcard) return domMatch;
  return domMatch || dowMatch;
}

// Local wall-clock enumeration; 8 years of days covers every date a 5-field expression can
// name, Feb 29 included.
export function nextFireTimes(cron: string, count = 5, from: Date = new Date()): Date[] {
  const parsed = parseCronExpression(cron);
  if (parsed === null || count <= 0) return [];
  const minutes = [...parsed.minute].sort((a, b) => a - b);
  const hours = [...parsed.hour].sort((a, b) => a - b);
  const startMs = from.getTime();
  const cursor = new Date(from.getFullYear(), from.getMonth(), from.getDate());
  const out: Date[] = [];

  for (let dayOffset = 0; dayOffset < 366 * 8; dayOffset += 1) {
    const date = new Date(cursor.getFullYear(), cursor.getMonth(), cursor.getDate() + dayOffset);
    if (!parsed.month.has(date.getMonth() + 1) || !dayMatches(parsed, date)) continue;
    for (const hour of hours) {
      for (const minute of minutes) {
        const fire = new Date(date.getFullYear(), date.getMonth(), date.getDate(), hour, minute, 0, 0);
        if (fire.getTime() <= startMs) continue;
        out.push(fire);
        if (out.length >= count) return out;
      }
    }
  }
  return out;
}

import { describe, expect, it } from 'vitest';
// @ts-expect-error -- plain .mjs generator, no types
import { buildRows } from '../data/augment.mjs';

interface Row {
  cron: string;
  bucket: string;
  phrasings: string[];
}

const rows: Row[] = buildRows();
const all = rows.flatMap((r) => r.phrasings.map((p) => ({ text: p, cron: r.cron, bucket: r.bucket })));

describe('augment', () => {
  it('is deterministic', () => {
    expect(buildRows()).toEqual(rows);
  });

  it('spells every ordinal correctly', () => {
    const suffix = (n: number) => {
      if (n % 100 >= 11 && n % 100 <= 13) return 'th';
      return ({ 1: 'st', 2: 'nd', 3: 'rd' })[n % 10] ?? 'th';
    };
    const wrong: string[] = [];
    for (const p of all) {
      for (const [, digits, suf] of p.text.matchAll(/\b(\d+)(st|nd|rd|th)\b/g)) {
        if (suf !== suffix(Number(digits))) wrong.push(p.text);
      }
    }
    expect(wrong).toEqual([]);
  });

  it('never emits the sub-minute sense', () => {
    expect(all.filter((p) => /\bseconds\b|\b\d+\s*seconds?\b/i.test(p.text))).toEqual([]);
  });

  it('teaches "second" as an ordinal and as a step, with distinct labels', () => {
    const ordinal = all.filter((p) => /\bthe second\b/i.test(p.text));
    const step = all.filter((p) => /\bevery second\b/i.test(p.text));
    expect(ordinal.length).toBeGreaterThan(0);
    expect(step.length).toBeGreaterThan(0);
    for (const p of ordinal) expect(p.cron.split(' ')[2]).toBe('2');
    for (const p of step) expect(p.cron).toMatch(/\*\/2/);
  });

  // The model was copying the day digit into the hour ("15th of the month" -> "0 15 15 * *");
  // a day with no stated time must pin the hour to 0 or that correlation survives.
  it('pins an unstated time to midnight for every day of the month', () => {
    const seen = new Set<number>();
    for (const r of rows.filter((x) => x.bucket === 'aug-dom-notime')) {
      const [minute, hour, dom] = r.cron.split(' ');
      expect(minute).toBe('0');
      expect(hour).toBe('0');
      seen.add(Number(dom));
    }
    expect([...seen].sort((a, b) => a - b)).toEqual(Array.from({ length: 31 }, (_, i) => i + 1));
  });

  it('covers all four spellings of a day number', () => {
    const second = rows.find((r) => r.cron === '0 0 2 * *');
    expect(second).toBeDefined();
    const joined = second!.phrasings.join(' | ');
    for (const form of ['the 2nd', 'the second', 'day 2', 'day two']) {
      expect(joined).toContain(form);
    }
  });
});

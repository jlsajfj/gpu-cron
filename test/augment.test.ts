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

  // "second" carries three senses and each must reach the model with its own label:
  // the day-of-month ordinal, the minute ordinal, and the step.
  it('teaches every sense of "second" with a distinct label', () => {
    const senses = {
      dom: all.filter((p) => /\bthe second\b/i.test(p.text) && !/minute/i.test(p.text)),
      minute: all.filter((p) => /\bthe second minute\b/i.test(p.text)),
      step: all.filter((p) => /\bevery second\b/i.test(p.text)),
    };
    for (const [name, hits] of Object.entries(senses)) {
      expect(hits.length, `no phrasings for the ${name} sense`).toBeGreaterThan(0);
    }
    // A list phrasing ("the second and sixteenth") puts 2 among the day values, not alone.
    const field = (cron: string, i: number) => (cron.split(' ')[i] ?? '').split(',');
    for (const p of senses.dom) expect(field(p.cron, 2)).toContain('2');
    for (const p of senses.minute) expect(field(p.cron, 0)).toContain('2');
    for (const p of senses.step) expect(p.cron).toMatch(/\*\/2/);
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

  // The sampler only ever emitted {0,15,30,45}; 53 of 60 minute values were absent and the
  // model snapped arbitrary minutes to the nearest tidy one.
  it('covers every minute value', () => {
    const minutes = new Set<number>();
    for (const r of rows.filter((x) => x.bucket === 'aug-time')) {
      const m = Number(r.cron.split(' ')[0]);
      if (Number.isInteger(m)) minutes.add(m);
    }
    expect(minutes.size).toBe(60);
  });

  it('teaches the fraction words', () => {
    for (const [phrase, minute] of [['quarter past', 15], ['half past', 30], ['quarter to', 45]] as const) {
      const hits = all.filter((p) => p.text.includes(phrase));
      expect(hits.length).toBeGreaterThan(0);
      for (const h of hits) expect(Number(h.cron.split(' ')[0])).toBe(minute);
    }
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

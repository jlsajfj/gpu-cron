import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import * as api from '../src/index.js';

// dist/index.d.ts is hand-written, so a new export reaches JS users but stays invisible to
// TypeScript ones unless someone remembers both files. isAvailable() nearly shipped that way.
const dts = readFileSync(new URL('../scripts/index.d.ts', import.meta.url), 'utf8');
const declared = [...dts.matchAll(/^export declare (?:function|class) (\w+)/gm)].map((m) => m[1]);

describe('public API', () => {
  it('exports exactly what the declarations promise', () => {
    expect(Object.keys(api).sort()).toEqual([...declared].sort());
  });

  it('isAvailable resolves false here rather than throwing', async () => {
    await expect(api.isAvailable()).resolves.toBe(false);
  });

  // React's use() and Suspense caches key on promise identity; a fresh promise per render
  // suspends forever. This is a load-bearing property of the API, not an implementation detail.
  it('returns the same promise object every call', async () => {
    const first = api.isAvailable();
    expect(api.isAvailable()).toBe(first);
    await first;
    expect(api.isAvailable()).toBe(first);
  });

  it('parse rejects with a CronError when there is no backend', async () => {
    await expect(api.parse('every day at 9am')).rejects.toBeInstanceOf(api.CronError);
  });
});

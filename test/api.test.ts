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

  it('parse rejects with a typed error you can branch on', async () => {
    await expect(api.parse('every day at 9am')).rejects.toBeInstanceOf(api.NoWebGpuError);
    await expect(api.parse('every day at 9am')).rejects.toBeInstanceOf(api.CronError);
  });

  // The pre-check and the throw must agree, or a tooltip could disagree with an error toast.
  it('hands back the same error instance parse throws', async () => {
    await api.isAvailable();
    const caught = await api.parse('every day at 9am').catch((e) => e);
    expect(api.unavailable()).toBe(caught);
  });

  it('every failure class descends from CronError', () => {
    for (const cls of [api.NoWebGpuError, api.NoModelError, api.InferenceFailedError, api.UngrammaticalError]) {
      expect(Object.create(cls.prototype)).toBeInstanceOf(api.CronError);
    }
  });
});

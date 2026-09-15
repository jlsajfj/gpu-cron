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

  it('parse rejects with a branchable reason when there is no backend', async () => {
    await expect(api.parse('every day at 9am')).rejects.toMatchObject({
      name: 'CronError',
      reason: 'no-webgpu',
    });
  });

  it('declares every reason the runtime can latch', () => {
    for (const reason of ['no-webgpu', 'no-model', 'inference-failed', 'ungrammatical']) {
      expect(dts).toContain(`'${reason}'`);
    }
  });
});

// Pins the hand-written forward pass to the numpy reference in export/export_js.py.
//
// The demo has no ONNX runtime and no autodiff to fall back on, so a wrong matmul here is a
// silently wrong cron rather than a crash. export_js.py already checks that the numpy
// reference matches torch; this closes the chain by checking that the TypeScript matches the
// reference on the same frozen weights.

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { loadModel, logitsFor, type ModelManifest } from '../src/forward.js';

const FIXTURE = new URL('./fixtures/', import.meta.url);
const bytes = new Uint8Array(readFileSync(new URL('tiny.bin', FIXTURE)));
const manifest = JSON.parse(readFileSync(new URL('tiny.json', FIXTURE), 'utf8')) as ModelManifest;
const reference = JSON.parse(readFileSync(new URL('tiny-reference.json', FIXTURE), 'utf8')) as {
  n_head: number;
  cases: { ids: number[]; logits: number[] }[];
};

describe('forward pass', () => {
  const model = loadModel(bytes, manifest);

  it('loads every tensor the manifest declares', () => {
    expect(manifest.tensors).toHaveLength(2 + 6 * manifest.n_layer + 1);
    expect(model.tok).toHaveLength(manifest.vocab * manifest.d_model);
    expect(model.blocks).toHaveLength(manifest.n_layer);
    expect(model.blocks[0]?.qkv).toHaveLength(3 * manifest.d_model * manifest.d_model);
  });

  it('reproduces the numpy reference for every fixture case', () => {
    expect(reference.cases.length).toBeGreaterThan(0);
    for (const { ids, logits } of reference.cases) {
      const got = logitsFor(model, ids);
      expect(got).toHaveLength(logits.length);
      let worst = 0;
      for (let i = 0; i < logits.length; i += 1) {
        worst = Math.max(worst, Math.abs((got[i] as number) - (logits[i] as number)));
      }
      // The reference is rounded to 5 decimals on the way out, so this is the rounding floor.
      expect(worst).toBeLessThan(1e-4);
    }
  });

  it('is deterministic and unaffected by logit-level ties in call order', () => {
    const ids = reference.cases[0]?.ids as number[];
    expect(Array.from(logitsFor(model, ids))).toEqual(Array.from(logitsFor(model, ids)));
  });

  it('gives the last position only, not the whole sequence', () => {
    const ids = reference.cases[0]?.ids as number[];
    expect(logitsFor(model, ids)).toHaveLength(manifest.vocab);
  });
});

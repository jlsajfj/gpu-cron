// Defines window.makeLogits for test/gpu-conformance.html: the WGSL forward pass from
// src/gpu.ts, behind the contract the page documents. Bundled to test/gpu-bundle.js by
// scripts/build.mjs, because the page itself runs no bundler.

import { loadGpuModel } from '../src/gpu.js';
import type { ModelManifest } from '../src/weights.js';

declare global {
  interface Window {
    makeLogits?: (
      bytes: Uint8Array,
      manifest: ModelManifest,
    ) => (ids: number[]) => Promise<Float32Array>;
  }
}

window.makeLogits = (bytes: Uint8Array, manifest: ModelManifest) => {
  const ready = loadGpuModel(bytes, manifest);
  return (ids: number[]) => ready.then((handle) => handle.logits(ids, 0));
};

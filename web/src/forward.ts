// The whole model, in plain TypeScript.
//
// ONNX Runtime Web is ~26 MB of wasm — more than this model, its weights and its dataset
// put together. For a few hundred thousand parameters the arithmetic is small enough that
// typed-array loops beat the download, so there is no runtime here at all: the weights are
// raw int8 bytes plus a manifest, and this file is the forward pass.
//
// This mirrors export/export_js.py's numpy reference function structurally, so if the two
// ever disagree it is a bug in one of them rather than a difference in how the arithmetic
// was grouped. export_js.py prints the max logit delta against torch on export.

import type { LogitsFn } from './decode.js';

export interface TensorEntry {
  name: string;
  kind: 'i8' | 'f32';
  shape: number[];
  offset: number;
  scaleOffset?: number;
}

export interface ModelManifest {
  d_model: number;
  n_layer: number;
  n_head: number;
  d_ff: number;
  max_len: number;
  vocab: number;
  tensors: TensorEntry[];
}

interface Block {
  qkv: Float32Array;
  proj: Float32Array;
  fc: Float32Array;
  mlpProj: Float32Array;
  norm1: Float32Array;
  norm2: Float32Array;
}

export interface Model {
  manifest: ModelManifest;
  tok: Float32Array;
  pos: Float32Array;
  blocks: Block[];
  normF: Float32Array;
}

const EPS = 1e-6;

/** Read a 2-D tensor, dequantizing only if it was stored as int8. */
function matrix(bytes: Uint8Array, entry: TensorEntry): Float32Array {
  if (entry.kind === 'f32') {
    const [rows, cols] = entry.shape as [number, number];
    return new Float32Array(bytes.buffer, bytes.byteOffset + entry.offset, rows * cols).slice();
  }
  const [rows, cols] = entry.shape as [number, number];
  const q = new Int8Array(bytes.buffer, bytes.byteOffset + entry.offset, rows * cols);
  const scales = new Float32Array(
    bytes.buffer,
    bytes.byteOffset + (entry.scaleOffset as number),
    rows,
  );
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    const scale = scales[r] as number;
    const base = r * cols;
    for (let c = 0; c < cols; c += 1) out[base + c] = (q[base + c] as number) * scale;
  }
  return out;
}

function floats(bytes: Uint8Array, entry: TensorEntry): Float32Array {
  const n = entry.shape.reduce((a, b) => a * b, 1);
  return new Float32Array(bytes.buffer, bytes.byteOffset + entry.offset, n).slice();
}

export function loadModel(bytes: Uint8Array, manifest: ModelManifest): Model {
  const byName = new Map(manifest.tensors.map((t) => [t.name, t]));
  const get = (name: string): TensorEntry => {
    const entry = byName.get(name);
    if (entry === undefined) throw new Error(`manifest is missing ${name}`);
    return entry;
  };
  const blocks: Block[] = [];
  for (let layer = 0; layer < manifest.n_layer; layer += 1) {
    const p = `blocks.${layer}.`;
    blocks.push({
      qkv: matrix(bytes, get(`${p}attn.qkv.weight`)),
      proj: matrix(bytes, get(`${p}attn.proj.weight`)),
      fc: matrix(bytes, get(`${p}mlp.fc.weight`)),
      mlpProj: matrix(bytes, get(`${p}mlp.proj.weight`)),
      norm1: floats(bytes, get(`${p}norm1.weight`)),
      norm2: floats(bytes, get(`${p}norm2.weight`)),
    });
  }
  return {
    manifest,
    tok: matrix(bytes, get('tok.weight')),
    pos: matrix(bytes, get('pos.weight')),
    blocks,
    normF: floats(bytes, get('norm_f.weight')),
  };
}

/** y = x @ Wᵀ, with W row-major (out, in). */
function project(x: Float32Array, rows: number, W: Float32Array, out: number): Float32Array {
  const cols = W.length / out;
  const y = new Float32Array(rows * out);
  for (let r = 0; r < rows; r += 1) {
    const xb = r * cols;
    const yb = r * out;
    for (let o = 0; o < out; o += 1) {
      const wb = o * cols;
      let sum = 0;
      for (let c = 0; c < cols; c += 1) sum += (x[xb + c] as number) * (W[wb + c] as number);
      y[yb + o] = sum;
    }
  }
  return y;
}

function rmsNorm(x: Float32Array, weight: Float32Array): Float32Array {
  const d = weight.length;
  const rows = x.length / d;
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r += 1) {
    const base = r * d;
    let sum = 0;
    for (let c = 0; c < d; c += 1) {
      const v = x[base + c] as number;
      sum += v * v;
    }
    const inv = 1 / Math.sqrt(sum / d + EPS);
    for (let c = 0; c < d; c += 1) {
      out[base + c] = (x[base + c] as number) * inv * (weight[c] as number);
    }
  }
  return out;
}

function gelu(x: Float32Array): Float32Array {
  const c = Math.sqrt(2 / Math.PI);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i += 1) {
    const v = x[i] as number;
    out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
  }
  return out;
}

function addInPlace(target: Float32Array, delta: Float32Array): void {
  for (let i = 0; i < target.length; i += 1) target[i] = (target[i] as number) + (delta[i] as number);
}

/** Logits for the last position of `ids`, over the full vocabulary. */
export function logitsFor(model: Model, ids: number[]): Float32Array {
  const { d_model: d, n_head: heads } = model.manifest;
  const headDim = d / heads;
  const t = ids.length;
  const scale = 1 / Math.sqrt(headDim);

  const x = new Float32Array(t * d);
  for (let p = 0; p < t; p += 1) {
    const token = ids[p] as number;
    for (let c = 0; c < d; c += 1) {
      x[p * d + c] = (model.tok[token * d + c] as number) + (model.pos[p * d + c] as number);
    }
  }

  const scores = new Float32Array(t);
  for (const block of model.blocks) {
    const h = rmsNorm(x, block.norm1);
    const qkv = project(h, t, block.qkv, 3 * d);
    const attn = new Float32Array(t * d);
    for (let head = 0; head < heads; head += 1) {
      const off = head * headDim;
      for (let i = 0; i < t; i += 1) {
        let best = -Infinity;
        for (let j = 0; j <= i; j += 1) {
          let dot = 0;
          for (let c = 0; c < headDim; c += 1) {
            dot += (qkv[i * 3 * d + off + c] as number) * (qkv[j * 3 * d + d + off + c] as number);
          }
          const v = dot * scale;
          scores[j] = v;
          if (v > best) best = v;
        }
        let total = 0;
        for (let j = 0; j <= i; j += 1) {
          const e = Math.exp((scores[j] as number) - best);
          scores[j] = e;
          total += e;
        }
        for (let c = 0; c < headDim; c += 1) {
          let sum = 0;
          for (let j = 0; j <= i; j += 1) {
            sum += (scores[j] as number) * (qkv[j * 3 * d + 2 * d + off + c] as number);
          }
          attn[i * d + off + c] = sum / total;
        }
      }
    }
    addInPlace(x, project(attn, t, block.proj, d));

    const h2 = rmsNorm(x, block.norm2);
    const expanded = gelu(project(h2, t, block.fc, model.manifest.d_ff));
    addInPlace(x, project(expanded, t, block.mlpProj, d));
  }

  const last = new Float32Array(d);
  const base = (t - 1) * d;
  for (let c = 0; c < d; c += 1) last[c] = x[base + c] as number;
  const normed = rmsNorm(last, model.normF);
  return project(normed, 1, model.tok, model.manifest.vocab); // tied embeddings
}

export function makeLogitsFn(model: Model): LogitsFn {
  return (ids: number[]) => Promise.resolve(logitsFor(model, ids));
}

// Reads the exported model: raw bytes plus a manifest, dequantized to float32.
//
// The arithmetic is WGSL (src/gpu.ts). This only turns the export into the tensors that get
// uploaded, so there is no second forward pass to keep in step with the first.

export interface TensorEntry {
  name: string;
  kind: 'i8' | 'f32';
  shape: number[];
  offset: number;
  scaleOffset?: number;
}

export interface ModelManifest {
  /** Recorded by the export for reference; the tensors carry their own kinds. */
  dtype?: string;
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

/** Read a 2-D tensor, dequantizing only if it was stored as int8. */
function matrix(bytes: Uint8Array, entry: TensorEntry): Float32Array {
  const [rows, cols] = entry.shape as [number, number];
  if (entry.kind === 'f32') {
    return new Float32Array(bytes.buffer, bytes.byteOffset + entry.offset, rows * cols).slice();
  }
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

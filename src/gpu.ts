// The same forward pass as forward.ts, as WGSL compute shaders on WebGPU.
//
// Ten small shaders rather than one big kernel: each dispatch is a few microseconds of work,
// and the arithmetic is off the JS thread entirely. Model dims are baked into the source at
// load — they are fixed for a given checkpoint — so the only per-token uniform is the
// sequence length.
//
// There is no CPU path: no adapter is an error, not a slower answer.
//
// The grammar mask stays on the CPU. One readback per token (vocab floats) is the price of
// constrained decoding, and reimplementing the automaton in WGSL would not pay for itself.

import type { LogitsFn } from './decode.js';
import { loadModel, type Model, type ModelManifest } from './weights.js';

const WG = 64;
const EPS = 1e-6;

export interface GpuHandle {
  adapter: string;
  logits: LogitsFn;
  destroy: () => void;
}

interface Step {
  pipeline: GPUComputePipeline;
  bind: GPUBindGroup;
  count: (t: number) => number;
}

const UNIFORM = `
struct Params { t: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(99) var<uniform> u: Params;
`;

function embedShader(d: number): string {
  return `
${UNIFORM}
@group(0) @binding(0) var<storage, read> tokW: array<f32>;
@group(0) @binding(1) var<storage, read> posW: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.t * ${d}u) { return; }
  let p = i / ${d}u;
  let c = i % ${d}u;
  x[i] = tokW[ids[p] * ${d}u + c] + posW[i];
}`;
}

function rmsnormShader(d: number): string {
  return `
${UNIFORM}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= u.t) { return; }
  let base = row * ${d}u;
  var sum = 0.0;
  for (var c = 0u; c < ${d}u; c = c + 1u) {
    let v = x[base + c];
    sum = sum + v * v;
  }
  let inv = 1.0 / sqrt(sum / ${d}.0 + ${EPS});
  for (var c = 0u; c < ${d}u; c = c + 1u) {
    out[base + c] = x[base + c] * inv * weight[c];
  }
}`;
}

// y[m][n] = sum_k x[m][k] * w[n][k]. The logits call is the same shape with m = 1 and the
// row pinned to the last position.
function matmulShader(n: number, k: number, rows: string, rowExpr: string): string {
  return `
${UNIFORM}
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= (${rows}) * ${n}u) { return; }
  let row = i / ${n}u;
  let col = i % ${n}u;
  let src = (${rowExpr}) * ${k}u;
  let dst = col * ${k}u;
  var sum = 0.0;
  for (var c = 0u; c < ${k}u; c = c + 1u) {
    sum = sum + x[src + c] * w[dst + c];
  }
  y[i] = sum;
}`;
}

// One invocation per (position, head). Three passes over the keys rather than a scores array:
// a per-invocation array of max_len floats spills out of registers, and recomputing the dot
// is cheaper than the spill at these sizes.
function attentionShader(d: number, heads: number, headDim: number): string {
  const threeD = 3 * d;
  return `
${UNIFORM}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> attn: array<f32>;

fn dot(qBase: u32, kBase: u32) -> f32 {
  var acc = 0.0;
  for (var c = 0u; c < ${headDim}u; c = c + 1u) {
    acc = acc + qkv[qBase + c] * qkv[kBase + c];
  }
  return acc;
}

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x / ${heads}u;
  let head = gid.x % ${heads}u;
  if (i >= u.t) { return; }
  let off = head * ${headDim}u;
  let qBase = i * ${threeD}u + off;
  let scale = 1.0 / sqrt(${headDim}.0);

  var best = -1e30;
  for (var j = 0u; j <= i; j = j + 1u) {
    best = max(best, dot(qBase, j * ${threeD}u + ${d}u + off) * scale);
  }
  var total = 0.0;
  for (var j = 0u; j <= i; j = j + 1u) {
    total = total + exp(dot(qBase, j * ${threeD}u + ${d}u + off) * scale - best);
  }
  for (var c = 0u; c < ${headDim}u; c = c + 1u) {
    var acc = 0.0;
    for (var j = 0u; j <= i; j = j + 1u) {
      let weight = exp(dot(qBase, j * ${threeD}u + ${d}u + off) * scale - best) / total;
      acc = acc + weight * qkv[j * ${threeD}u + ${2 * d}u + off + c];
    }
    attn[i * ${d}u + off + c] = acc;
  }
}`;
}

function geluShader(ff: number): string {
  return `
${UNIFORM}
@group(0) @binding(0) var<storage, read_write> mlp: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.t * ${ff}u) { return; }
  let v = mlp[i];
  let u = 0.7978845608 * (v + 0.044715 * v * v * v);
  // A naive exp-based tanh returns NaN once its argument saturates, and v*v*v overflows f32 for
  // large activations, so clamp where f32 tanh is already exactly +/-1. Without this the real
  // checkpoint decodes to all-NaN logits on SwiftShader while a small random model is fine.
  let t = clamp(u, -15.0, 15.0);
  mlp[i] = 0.5 * v * (1.0 + tanh(t));
}`;
}

function addShader(d: number): string {
  return `
${UNIFORM}
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.t * ${d}u) { return; }
  x[i] = x[i] + delta[i];
}`;
}

function withUniform(code: string, arity: number): string {
  return code.replace('@binding(99)', `@binding(${arity})`);
}

type Mode = 'read-only-storage' | 'storage';

// A layout binding's type must match the shader's access mode exactly, so it is read back out
// of the source rather than declared twice.
function storageModes(code: string, arity: number): Mode[] {
  const modes: Mode[] = Array.from({ length: arity }, () => 'read-only-storage');
  const pattern = /@group\(0\) @binding\((\d+)\) var<storage, (read_write|read)>/g;
  for (const match of code.matchAll(pattern)) {
    const index = Number(match[1]);
    if (index < arity) modes[index] = match[2] === 'read_write' ? 'storage' : 'read-only-storage';
  }
  return modes;
}

export class NoWebGpuError extends Error {
  override name = 'NoWebGpuError';
}

export async function loadGpuModel(bytes: Uint8Array, manifest: ModelManifest): Promise<GpuHandle> {
  if (typeof navigator === 'undefined' || navigator.gpu === undefined) {
    throw new NoWebGpuError('This browser has no WebGPU (navigator.gpu is undefined).');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) {
    throw new NoWebGpuError('No WebGPU adapter is available on this device.');
  }
  const device = await adapter.requestDevice();
  const info = (adapter as unknown as { info?: { description?: string } }).info;

  const model: Model = loadModel(bytes, manifest);
  const { d_model: d, n_layer: layers, n_head: heads, d_ff: ff, vocab } = manifest;
  const headDim = d / heads;
  const threeD = 3 * d;

  // Pipeline creation reports a bad layout or binding asynchronously, and the invalid pipeline
  // only surfaces later as a dropped dispatch — so the whole setup is checked before use.
  device.pushErrorScope('validation');

  const owned: GPUBuffer[] = [];
  const storage = (data: Float32Array): GPUBuffer => {
    const buffer = device.createBuffer({
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buffer, 0, data as unknown as GPUAllowSharedBufferSource);
    owned.push(buffer);
    return buffer;
  };
  const scratch = (floats: number): GPUBuffer => {
    const buffer = device.createBuffer({
      size: floats * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    owned.push(buffer);
    return buffer;
  };

  const tokW = storage(model.tok);
  const posW = storage(model.pos);
  const ids = scratch(manifest.max_len);
  const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  owned.push(uniform);

  const x = scratch(manifest.max_len * d);
  const xnorm = scratch(manifest.max_len * d);
  const qkv = scratch(manifest.max_len * threeD);
  const attn = scratch(manifest.max_len * d);
  const mlp = scratch(manifest.max_len * ff);
  const proj = scratch(manifest.max_len * d);
  const logitsBuf = device.createBuffer({
    size: vocab * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const staging = device.createBuffer({
    size: vocab * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  owned.push(logitsBuf, staging);

  const layouts = new Map<string, GPUBindGroupLayout>();
  const layoutFor = (modes: Mode[]): GPUBindGroupLayout => {
    const key = modes.join('|');
    const cached = layouts.get(key);
    if (cached !== undefined) return cached;
    const created = device.createBindGroupLayout({
      entries: [
        ...modes.map((type, binding) => ({
          binding,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type },
        })),
        {
          binding: modes.length,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' as const },
        },
      ],
    });
    layouts.set(key, created);
    return created;
  };

  const pipelines = new Map<string, GPUComputePipeline>();
  // A shader that fails to compile produces a pipeline that silently dispatches nothing, and
  // zeroed logits decode to a confident wrong answer — so both are errors here, loudly.
  const step = async (
    name: string,
    code: string,
    buffers: GPUBuffer[],
    count: (t: number) => number,
  ): Promise<Step> => {
    const arity = buffers.length;
    const modes = storageModes(code, arity);
    let pipeline = pipelines.get(name);
    if (pipeline === undefined) {
      const module = device.createShaderModule({ code: withUniform(code, arity) });
      const info = await module.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === 'error');
      if (errors.length > 0) {
        const detail = errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ');
        throw new Error(`${name} shader: ${detail}`);
      }
      pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layoutFor(modes)] }),
        compute: { module, entryPoint: 'main' },
      });
      pipelines.set(name, pipeline);
    }
    const bind = device.createBindGroup({
      layout: layoutFor(modes),
      entries: [
        ...buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        { binding: arity, resource: { buffer: uniform } },
      ],
    });
    return { pipeline, bind, count };
  };

  const steps: Step[] = [
    await step('embed', embedShader(d), [tokW, posW, ids, x], (t) => t * d),
  ];
  for (let layer = 0; layer < layers; layer += 1) {
    const block = model.blocks[layer] as NonNullable<Model['blocks'][number]>;
    const qkvW = storage(block.qkv);
    const projW = storage(block.proj);
    const fcW = storage(block.fc);
    const mlpProjW = storage(block.mlpProj);
    const norm1 = storage(block.norm1);
    const norm2 = storage(block.norm2);
    steps.push(
      await step('rmsnorm', rmsnormShader(d), [x, norm1, xnorm], (t) => t),
      await step('qkv', matmulShader(threeD, d, 'u.t', 'row'), [xnorm, qkvW, qkv], (t) => t * threeD),
      await step('attention', attentionShader(d, heads, headDim), [qkv, attn], (t) => t * heads),
      await step('proj', matmulShader(d, d, 'u.t', 'row'), [attn, projW, proj], (t) => t * d),
      await step('add', addShader(d), [x, proj], (t) => t * d),
      await step('rmsnorm', rmsnormShader(d), [x, norm2, xnorm], (t) => t),
      await step('fc', matmulShader(ff, d, 'u.t', 'row'), [xnorm, fcW, mlp], (t) => t * ff),
      await step('gelu', geluShader(ff), [mlp], (t) => t * ff),
      await step('mlpProj', matmulShader(d, ff, 'u.t', 'row'), [mlp, mlpProjW, proj], (t) => t * d),
      await step('add', addShader(d), [x, proj], (t) => t * d),
    );
  }
  const normF = storage(model.normF);
  steps.push(
    await step('rmsnorm', rmsnormShader(d), [x, normF, xnorm], (t) => t),
    await step('logits', matmulShader(vocab, d, '1u', 'u.t - 1u'), [xnorm, tokW, logitsBuf], () => vocab),
  );

  const idScratch = new Uint32Array(manifest.max_len);
  const logits = async (prompt: number[]): Promise<Float32Array> => {
    const t = prompt.length;
    idScratch.set(prompt, 0);
    device.queue.writeBuffer(ids, 0, idScratch, 0, t);
    device.queue.writeBuffer(uniform, 0, new Uint32Array([t, 0, 0, 0]));

    // A dropped dispatch leaves the logits buffer reading zero, which decodes to a confident
    // wrong answer rather than an error: surface the validation failure instead.
    device.pushErrorScope('validation');
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    let bound: GPUComputePipeline | null = null;
    for (const step of steps) {
      if (step.pipeline !== bound) {
        pass.setPipeline(step.pipeline);
        bound = step.pipeline;
      }
      pass.setBindGroup(0, step.bind);
      pass.dispatchWorkgroups(Math.ceil(step.count(t) / WG));
    }
    pass.end();
    encoder.copyBufferToBuffer(logitsBuf, 0, staging, 0, staging.size);
    device.queue.submit([encoder.finish()]);

    const failure = await device.popErrorScope();
    if (failure !== null) throw new Error(`webgpu rejected the pass: ${failure.message}`);

    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return out;
  };

  const setupFailure = await device.popErrorScope();
  if (setupFailure !== null) throw new Error(`gpu setup: ${setupFailure.message}`);

  return {
    adapter: info?.description || 'webgpu',
    logits,
    destroy: () => {
      for (const buffer of owned) buffer.destroy();
    },
  };
}

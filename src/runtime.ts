// Loads the inlined weights onto the GPU and hands back a logits function.
//
// There is no second backend. A browser without WebGPU gets a clear error rather than a
// different, slower answer, and there is no inference runtime to download either way.

import { loadGpuModel, NoWebGpuError } from './gpu.js';
import { MANIFEST, MODEL_BASE64, MODEL_BYTES, MODEL_NAME, MODEL_PRESENT } from './weights.generated.js';

export type FailureReason = 'no-model' | 'no-webgpu' | 'inference-failed';

export interface Runtime {
  adapter: string;
  modelBytes: number;
  modelName: string;
  params: number;
  logits: (ids: number[], position: number) => Promise<Float32Array>;
}

export interface RuntimeFailure {
  reason: FailureReason;
  message: string;
}

export type RuntimeResult = { ok: true; runtime: Runtime } | ({ ok: false } & RuntimeFailure);

let latched: RuntimeFailure | null = null;

function latch(failure: RuntimeFailure): RuntimeFailure {
  latched ??= failure;
  return latched;
}

export function modelSizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function paramLabel(params: number): string {
  return params < 1000 ? `${params}` : `${(params / 1000).toFixed(0)}k`;
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export async function loadRuntime(): Promise<RuntimeResult> {
  if (latched !== null) return { ok: false, ...latched };

  if (!MODEL_PRESENT || MODEL_BASE64.length === 0) {
    return {
      ok: false,
      ...latch({
        reason: 'no-model',
        message: `No model at ${MODEL_NAME}. Export one with export/export_js.py and rebuild.`,
      }),
    };
  }

  try {
    const bytes = base64ToBytes(MODEL_BASE64);
    const params = MANIFEST.tensors.reduce(
      (sum, entry) => sum + entry.shape.reduce((a, b) => a * b, 1),
      0,
    );

    const gpu = await loadGpuModel(bytes, MANIFEST);
    return {
      ok: true,
      runtime: {
        adapter: gpu.adapter,
        modelBytes: MODEL_BYTES,
        modelName: MODEL_NAME,
        params,
        logits: gpu.logits,
      },
    };
  } catch (error) {
    if (error instanceof NoWebGpuError) {
      return { ok: false, ...latch({ reason: 'no-webgpu', message: error.message }) };
    }
    return {
      ok: false,
      ...latch({
        reason: 'inference-failed',
        message: `The model is in this build but would not load: ${
          error instanceof Error ? error.message : String(error)
        }`,
      }),
    };
  }
}

export function latchInferenceFailure(error: unknown): RuntimeFailure {
  return latch({
    reason: 'inference-failed',
    message: error instanceof Error ? error.message : String(error),
  });
}

export function latchedFailure(): RuntimeFailure | null {
  return latched;
}

// Picks a backend and hands back a logits function. The weights are inlined at build time,
// so there is nothing to fetch and no inference runtime to download — WebGPU runs the show
// when it exists and the plain-TypeScript forward pass covers everything else.

import { loadModel, makeLogitsFn } from './forward.js';
import { loadGpuModel } from './gpu.js';
import { MANIFEST, MODEL_BASE64, MODEL_BYTES, MODEL_NAME, MODEL_PRESENT } from './weights.generated.js';

export type FailureReason = 'no-model' | 'inference-failed';

export interface Runtime {
  backend: 'webgpu' | 'cpu';
  adapter: string | null;
  modelBytes: number;
  modelName: string;
  params: number;
  logits: ReturnType<typeof makeLogitsFn>;
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
    const logits = gpu?.logits ?? makeLogitsFn(loadModel(bytes, MANIFEST));
    return {
      ok: true,
      runtime: {
        backend: gpu === null ? 'cpu' : 'webgpu',
        adapter: gpu?.adapter ?? null,
        modelBytes: MODEL_BYTES,
        modelName: MODEL_NAME,
        params,
        logits,
      },
    };
  } catch (error) {
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

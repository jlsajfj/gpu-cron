// Loads the inlined weights and exposes a logits function.
//
// There is no inference runtime here on purpose. ONNX Runtime Web is ~26 MB of wasm for a
// model of a few hundred thousand parameters, which is a download cost bigger than the
// thing it runs; web/src/forward.ts does the arithmetic on typed arrays instead, so the
// page needs no GPU and no wasm — it works the same everywhere.

import type { LogitsFn } from './decode.js';
import { loadModel, makeLogitsFn, type ModelManifest } from './forward.js';

export type FailureReason = 'no-model' | 'inference-failed';

export interface Runtime {
  modelBytes: number;
  modelName: string;
  params: number;
  logits: LogitsFn;
}

export interface RuntimeFailure {
  reason: FailureReason;
  message: string;
}

export type RuntimeResult = { ok: true; runtime: Runtime } | ({ ok: false } & RuntimeFailure);

interface WeightsModule {
  MODEL_BASE64: string;
  MANIFEST: ModelManifest;
  MODEL_BYTES: number;
  MODEL_PRESENT: boolean;
  MODEL_NAME: string;
}

const WEIGHTS_URL = 'weights.generated.js';

let latched: RuntimeFailure | null = null;

function latch(failure: RuntimeFailure): RuntimeFailure {
  latched ??= failure;
  return latched;
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function loadWeights(): Promise<WeightsModule | null> {
  try {
    const url = new URL(WEIGHTS_URL, import.meta.url).href;
    return (await import(url)) as WeightsModule;
  } catch {
    return null;
  }
}

export function modelSizeLabel(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function paramLabel(params: number): string {
  return params < 1000 ? `${params}` : `${(params / 1000).toFixed(0)}k`;
}

export async function loadRuntime(): Promise<RuntimeResult> {
  if (latched !== null) return { ok: false, ...latched };

  const weights = await loadWeights();
  if (weights === null || !weights.MODEL_PRESENT || weights.MODEL_BASE64.length === 0) {
    const where = weights === null ? WEIGHTS_URL : weights.MODEL_NAME;
    return {
      ok: false,
      ...latch({
        reason: 'no-model',
        message: `No model at ${where}. Export one with export/export_js.py and rebuild.`,
      }),
    };
  }

  try {
    const bytes = base64ToBytes(weights.MODEL_BASE64);
    const model = loadModel(bytes, weights.MANIFEST);
    const params = weights.MANIFEST.tensors.reduce(
      (sum, entry) => sum + entry.shape.reduce((a, b) => a * b, 1),
      0,
    );
    return {
      ok: true,
      runtime: {
        modelBytes: weights.MODEL_BYTES,
        modelName: weights.MODEL_NAME,
        params,
        logits: makeLogitsFn(model),
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

import * as ort from 'onnxruntime-web';
import type { LogitsFn } from './decode.js';
import { VOCAB_SIZE } from './tokenizer.js';

export type FailureReason = 'no-webgpu' | 'no-onnx' | 'inference-failed';

export type ExecutionProvider = 'webgpu' | 'wasm';

export interface Runtime {
  provider: ExecutionProvider;
  webgpuAvailable: boolean;
  modelBytes: number;
  modelName: string;
  logits: LogitsFn;
}

export interface RuntimeFailure {
  reason: FailureReason;
  message: string;
}

export type RuntimeResult = { ok: true; runtime: Runtime } | ({ ok: false } & RuntimeFailure);

interface WeightsModule {
  MODEL_BASE64: string;
  MODEL_BYTES: number;
  MODEL_PRESENT: boolean;
  MODEL_NAME: string;
}

interface GpuNavigator {
  requestAdapter(): Promise<unknown>;
}

const WEIGHTS_URL = 'weights.generated.js';

let latched: RuntimeFailure | null = null;

function latch(failure: RuntimeFailure): RuntimeFailure {
  latched ??= failure;
  return latched;
}

function gpuAvailable(): Promise<boolean> {
  const gpu = (navigator as unknown as { gpu?: GpuNavigator }).gpu;
  if (gpu === undefined) return Promise.resolve(false);
  return gpu
    .requestAdapter()
    .then((adapter) => adapter !== null)
    .catch(() => false);
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
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export async function loadRuntime(): Promise<RuntimeResult> {
  if (latched !== null) return { ok: false, ...latched };

  const weights = await loadWeights();
  if (weights === null || !weights.MODEL_PRESENT || weights.MODEL_BASE64.length === 0) {
    const where = weights === null ? WEIGHTS_URL : weights.MODEL_NAME;
    return {
      ok: false,
      ...latch({
        reason: 'no-onnx',
        message: `No quantized model at ${where}. Export one with export/export_onnx.py --int8 and rebuild.`,
      }),
    };
  }

  // No public API reports the provider a session settled on, so try WebGPU alone first and
  // fall back to the wasm build; whichever create() succeeds is the one actually running.
  const useWebgpu = await gpuAvailable();
  // Single threaded keeps the wasm build off SharedArrayBuffer, which needs COOP/COEP headers.
  ort.env.wasm.numThreads = 1;
  // build.mjs copies the wasm binaries into dist/ort/, so point ORT there, not beside app.js.
  ort.env.wasm.wasmPaths = new URL('ort/', import.meta.url).href;

  const bytes = base64ToBytes(weights.MODEL_BASE64);
  const attempts: ExecutionProvider[] = useWebgpu ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError: unknown = null;

  for (const provider of attempts) {
    try {
      const session = await ort.InferenceSession.create(bytes, { executionProviders: [provider] });
      const inputName = session.inputNames[0];
      const outputName = session.outputNames[0];
      if (inputName === undefined || outputName === undefined) {
        throw new Error('model has no input or output');
      }
      const logits: LogitsFn = async (ids) => {
        const failed = latched;
        if (failed !== null) throw new Error(failed.message);
        const input = new BigInt64Array(ids.length);
        for (let i = 0; i < ids.length; i += 1) input[i] = BigInt(ids[i] as number);
        // The graph returns last-position logits only; re-running the prefix beats a KV cache.
        const output = await session.run({ [inputName]: new ort.Tensor('int64', input, [1, ids.length]) });
        const data = output[outputName]?.data;
        if (!(data instanceof Float32Array) || data.length !== VOCAB_SIZE) {
          throw new Error(`unexpected logits shape from ${outputName}`);
        }
        return data;
      };
      return {
        ok: true,
        runtime: { provider, webgpuAvailable: useWebgpu, modelBytes: weights.MODEL_BYTES, modelName: weights.MODEL_NAME, logits },
      };
    } catch (error) {
      lastError = error;
    }
  }

  const detail = lastError instanceof Error ? lastError.message : String(lastError);
  return {
    ok: false,
    ...latch({
      // WebGPU missing is the headline only when the wasm fallback could not run either.
      reason: useWebgpu ? 'inference-failed' : 'no-webgpu',
      message: useWebgpu
        ? `WebGPU and the wasm fallback both failed to start the session: ${detail}`
        : `WebGPU is unavailable and the wasm fallback failed to start the session: ${detail}`,
    }),
  };
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

// Shipped to dist/index.d.ts. The surface is small enough to state by hand, and hand-written
// declarations are what keep the package one file with no internal type imports.
// Keep in step with src/index.ts — nothing checks that it matches.

export interface CronMatch {
  /** Always valid cron: the decoder cannot emit a token the grammar rejects. */
  expression: string;
  /** Upcoming fire times as ISO-8601 strings. */
  next: string[];
}

export interface ParseOptions {
  /** How many upcoming fire times to return. Defaults to 5. */
  count?: number;
}

export interface Backend {
  /** GPU adapter description, e.g. "Apple M2 Pro". */
  adapter: string;
  /** Parameter count of the loaded model. */
  params: number;
  /** Size of the inlined weights in bytes. */
  bytes: number;
}

/**
 * Why a call failed.
 *
 * - `no-webgpu` — no WebGPU adapter here. Not recoverable; there is no CPU fallback, and
 *   this is what importing the package in Node gives you.
 * - `no-model` — the build has no weights inlined. A packaging bug.
 * - `inference-failed` — WebGPU is present but the pipeline would not load, or a run threw.
 * - `ungrammatical` — the model could not finish a legal expression within the length cap.
 *   Input-specific: another phrasing may work.
 */
export type CronErrorReason = 'no-webgpu' | 'no-model' | 'inference-failed' | 'ungrammatical';

export declare class CronError extends Error {
  readonly reason: CronErrorReason;
}

/**
 * Whether this environment can run the model.
 *
 * Resolves `false` rather than throwing, so it is safe to call anywhere — including in
 * Node, where it is always `false`.
 *
 * The first call uploads the weights to the GPU. That work is shared with `parse()`, so
 * calling this first makes the first `parse()` warm instead of paying for the upload twice.
 */
export declare function isAvailable(): Promise<boolean>;

/**
 * Turn an English schedule description into a cron expression.
 *
 * The returned `expression` is always syntactically valid cron, but not necessarily the
 * schedule you meant — nonsense input yields a confident, valid, meaningless expression.
 *
 * Throws {@link CronError}; branch on its `reason`.
 */
export declare function parse(text: string, options?: ParseOptions): Promise<CronMatch>;

/** Which backend the model loaded on. Null until isAvailable() or parse() has loaded it. */
export declare function backend(): Backend | null;

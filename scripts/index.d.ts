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
 * Base class for every failure this package raises. Branch with `instanceof` on the
 * subclasses rather than matching on strings.
 */
export declare class CronError extends Error {}

/**
 * No WebGPU adapter here. Not recoverable: there is no CPU fallback, and this is what
 * importing the package in Node gives you.
 */
export declare class NoWebGpuError extends CronError {}

/** The build has no weights inlined. A packaging bug, not a runtime one. */
export declare class NoModelError extends CronError {}

/** WebGPU is present, but the pipeline would not load or a run threw. */
export declare class InferenceFailedError extends CronError {}

/**
 * The model could not finish a legal expression within the length cap. Unlike the others
 * this is input-specific and worth retrying: a different phrasing may work.
 */
export declare class UngrammaticalError extends CronError {}

/**
 * Whether this environment can run the model.
 *
 * Resolves `false` rather than throwing, so it is safe to call anywhere — including in
 * Node, where it is always `false`.
 *
 * The first call uploads the weights to the GPU. That work is shared with `parse()`, so
 * calling this first makes the first `parse()` warm instead of paying for the upload twice.
 *
 * Returns the same promise object every call, so it can be handed straight to React's
 * `use()` or any Suspense cache that keys on promise identity.
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

/**
 * The error explaining why the model is unavailable, or null if it is available or nothing
 * has checked yet. The same instance parse() would throw, without having to catch.
 *
 * isAvailable() tells you whether to disable your input; this tells you what to say.
 * Latched, so it is safe to read during render.
 */
export declare function unavailable(): CronError | null;

/** Which backend the model loaded on. Null until isAvailable() or parse() has loaded it. */
export declare function backend(): Backend | null;

// The public API: two functions. `isAvailable()` asks whether this environment can run the
// model; `parse()` turns English into cron. Everything else — adapter, weights, decoder,
// grammar automaton — is internal and set up once, on whichever of the two is called first.

import { defaultAutomaton, isWellFormed } from './automaton.js';
import { PROMPT_OVERHEAD } from './tokenizer.js';
import { nextFireTimes } from './cron.js';
import { decode } from './decode.js';
import { loadRuntime } from './runtime.js';

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

/** Anything that stops parse() from answering. */
export class CronError extends Error {
  override name = 'CronError';
}

export interface Backend {
  /** GPU adapter description, e.g. "Apple M2 Pro". */
  adapter: string;
  /** Parameter count of the loaded model. */
  params: number;
  /** Size of the inlined weights in bytes. */
  bytes: number;
}

let ready: ReturnType<typeof loadRuntime> | null = null;
let resolved: Backend | null = null;
let availability: Promise<boolean> | null = null;

// Both entry points share one load, so calling isAvailable() first makes parse() warm
// instead of doing the GPU upload twice.
async function load() {
  ready ??= loadRuntime();
  const result = await ready;
  if (result.ok) {
    resolved = {
      adapter: result.runtime.adapter,
      params: result.runtime.params,
      bytes: result.runtime.modelBytes,
    };
  }
  return result;
}

/**
 * Whether this environment can run the model.
 *
 * Resolves `false` rather than throwing, so it is safe to call anywhere — including in
 * Node, where it is always `false`. Use it to decide whether to render your UI at all:
 *
 * ```js
 * if (!(await isAvailable())) {
 *   showFallbackInput();   // a plain cron text field, say
 *   return;
 * }
 * ```
 *
 * The first call loads the weights onto the GPU, which is the expensive part (tens of
 * milliseconds). That work is shared with `parse()`, so calling this first makes the
 * first `parse()` fast rather than doing the work twice. Repeat calls are free.
 *
 * If you need to know *why* it is unavailable, call `parse()` and read `error.reason`.
 *
 * The same promise object is returned every call, so it works directly with React's
 * `use()` and with Suspense caches that key on identity:
 *
 * ```jsx
 * function Parser() {
 *   if (!use(isAvailable())) return <PlainCronInput />;
 *   return <ModelInput />;
 * }
 * ```
 */
export function isAvailable(): Promise<boolean> {
  // Deliberately not `async`: every call must return the SAME promise object. React's
  // `use()` and every Suspense cache key on promise identity, and a fresh promise per
  // render suspends forever.
  availability ??= load().then((result) => result.ok);
  return availability;
}

/**
 * Which backend the model loaded on, or `null` before anything has loaded it.
 *
 * Populated by the first successful `isAvailable()` or `parse()`. Diagnostics only.
 */
export function backend(): Backend | null {
  return resolved;
}

/**
 * Turn an English schedule description into a cron expression.
 *
 * ```js
 * await parse('every weekday at 9am');
 * // { expression: '0 9 * * 1-5', next: ['2026-01-01T09:00:00.000Z', ...] }
 * ```
 *
 * The returned `expression` is always syntactically valid cron — the decoder masks the
 * model's logits against a grammar automaton, so an invalid string is unreachable rather
 * than merely unlikely. It is **not** guaranteed to be the schedule you meant: this is a
 * 45k-parameter model, and there is no way for it to signal that it did not understand
 * you. Nonsense input yields a confident, valid, meaningless expression.
 *
 * Throws {@link CronError} with a `reason` — see {@link CronErrorReason}. Guard with
 * {@link isAvailable} if you would rather branch than catch.
 *
 * @param text  free-form English, e.g. `"every second tuesday at 9"`
 * @param options.count  how many upcoming fire times to return (default 5)
 */
export async function parse(text: string, options: ParseOptions = {}): Promise<CronMatch> {
  const runtime = await load();
  if (!runtime.ok) throw new CronError(runtime.message);

  const budget = runtime.runtime.maxLen - PROMPT_OVERHEAD - defaultAutomaton().grammar.maxLength - 1;
  if (text.length > budget) {
    throw new CronError(
      `prompt is ${text.length} characters; the model fits ${budget} alongside its answer`,
    );
  }

  const decoded = await decode(text, runtime.runtime.logits, { automaton: defaultAutomaton() });
  if (decoded.truncated || !isWellFormed(decoded.text)) {
    // Unreachable: the automaton cannot enter a state it is unable to close before the cap.
    // Not a parse outcome and not something a caller can act on — if it fires, this package
    // is broken, so it is an invariant failure rather than a branchable error class.
    throw new CronError(
      `human-cron invariant violated on ${JSON.stringify(text)} — this is a bug, please report it`,
    );
  }

  return {
    expression: decoded.text,
    next: nextFireTimes(decoded.text, options.count ?? 5).map((when) => when.toISOString()),
  };
}

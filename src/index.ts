// The public API, deliberately the same shape as gpu-lexer's: one async function in, plain
// data out. Everything else — adapter, weights, decoder, grammar automaton — is internal and
// set up once on first call.

import { defaultAutomaton, isWellFormed } from './automaton.js';
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

export class CronError extends Error {
  override name = 'CronError';
}

export interface Backend {
  runtime: 'webgpu' | 'cpu';
  adapter: string | null;
  params: number;
  bytes: number;
}

let ready: ReturnType<typeof loadRuntime> | null = null;
let resolved: Backend | null = null;

/** Which path the first parse() took. Null until something has been parsed. */
export function backend(): Backend | null {
  return resolved;
}

// TEMPORARY probe: exposes the raw logits fn so the browser and node paths can be diffed.
export async function parse(text: string, options: ParseOptions = {}): Promise<CronMatch> {
  ready ??= loadRuntime();
  const runtime = await ready;
  if (!runtime.ok) throw new CronError(runtime.message);

  resolved = {
    runtime: runtime.runtime.backend,
    adapter: runtime.runtime.adapter,
    params: runtime.runtime.params,
    bytes: runtime.runtime.modelBytes,
  };

  const decoded = await decode(text, runtime.runtime.logits, { automaton: defaultAutomaton() });
  if (decoded.truncated || !isWellFormed(decoded.text)) {
    throw new CronError(`could not finish ${JSON.stringify(text)} within the grammar`);
  }

  return {
    expression: decoded.text,
    next: nextFireTimes(decoded.text, options.count ?? 5).map((when) => when.toISOString()),
  };
}

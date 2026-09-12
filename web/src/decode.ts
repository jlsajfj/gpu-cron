import { EOS as EOS_CHAR, defaultAutomaton, type CronAutomaton, type State } from './automaton.js';
import { EOS_TOKEN_ID, VOCAB_SIZE, encodePrompt } from './tokenizer.js';

export type LogitsFn = (ids: number[], position: number) => Promise<Float32Array>;

export interface DecodeStats {
  truncated: number;
  decodes: number;
}

export interface DecodeResult {
  text: string;
  tokens: number[];
  truncated: boolean;
  steps: number;
  tokenMillis: number[];
  totalMillis: number;
}

export interface DecodeOptions {
  maxNew?: number;
  automaton?: CronAutomaton;
  stats?: DecodeStats;
  now?: () => number;
}

export function newStats(): DecodeStats {
  return { truncated: 0, decodes: 0 };
}

export function allowedIds(automaton: CronAutomaton, state: State): Set<number> {
  const out = new Set<number>();
  for (const ch of automaton.allowed(state)) {
    out.add(ch === EOS_CHAR ? EOS_TOKEN_ID : ch.charCodeAt(0));
  }
  return out;
}

export function maskLogits(logits: Float32Array, allowed: ReadonlySet<number>): Float32Array {
  const masked = new Float32Array(logits);
  for (let id = 0; id < masked.length; id += 1) {
    if (!allowed.has(id)) masked[id] = -Infinity;
  }
  return masked;
}

export function argmax(logits: Float32Array): number {
  let best = 0;
  let bestValue = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = logits[i] as number;
    if (v > bestValue) {
      bestValue = v;
      best = i;
    }
  }
  return best;
}

export async function decode(
  text: string,
  logits: LogitsFn,
  options: DecodeOptions = {},
): Promise<DecodeResult> {
  const automaton = options.automaton ?? defaultAutomaton();
  // Each legal move adds one character, capped at maxLength by the budget gate; the +1 is EOS.
  const maxNew = options.maxNew ?? automaton.grammar.maxLength + 1;
  const now = options.now ?? (() => performance.now());

  const promptIds = encodePrompt(text);
  const ids = [...promptIds];
  let state = automaton.start();
  const emitted: string[] = [];
  const tokenMillis: number[] = [];
  const startedAt = now();
  let truncated = true;

  for (let step = 0; step < maxNew; step += 1) {
    const stepStart = now();
    const raw = await logits(ids, ids.length);
    if (raw.length !== VOCAB_SIZE) {
      throw new Error(`logits width ${raw.length}, expected ${VOCAB_SIZE}`);
    }
    const allowed = allowedIds(automaton, state);
    // The budget gate keeps allowed() non-empty at every reachable state; empty is a bug.
    if (allowed.size === 0) {
      throw new Error(`automaton allowed no move from field ${state.field} ${JSON.stringify(state.text)}`);
    }
    const next = argmax(maskLogits(raw, allowed));
    tokenMillis.push(now() - stepStart);

    if (next === EOS_TOKEN_ID) {
      truncated = false;
      break;
    }
    const ch = String.fromCharCode(next);
    const advanced = automaton.advance(state, ch);
    if (advanced === null) {
      throw new Error(`automaton rejected its own allowed character ${JSON.stringify(ch)}`);
    }
    state = advanced;
    emitted.push(ch);
    ids.push(next);
  }

  if (options.stats) {
    options.stats.decodes += 1;
    if (truncated) options.stats.truncated += 1;
  }

  return {
    text: emitted.join(''),
    tokens: ids.slice(promptIds.length),
    truncated,
    steps: emitted.length,
    tokenMillis,
    totalMillis: now() - startedAt,
  };
}

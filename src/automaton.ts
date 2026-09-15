import { defaultGrammar, type FieldSpec, type Grammar } from './grammar.js';

export const EOS = '\u0000';

export const ALPHABET_CHARS = defaultGrammar().alphabet.replace(/ /g, '');

export interface State {
  readonly field: number;
  readonly text: string;
  readonly total: number;
}

export interface FieldState {
  readonly isPrefix: boolean;
  readonly canEnd: boolean;
}

function isDigit(c: string): boolean {
  return c >= '0' && c <= '9';
}

const leadingCache = new Map<string, boolean>();

function leadingOk(digit: string, lo: number, hi: number): boolean {
  const key = `${digit}:${lo}:${hi}`;
  const hit = leadingCache.get(key);
  if (hit !== undefined) return hit;
  let ok = false;
  for (let v = lo; v <= hi && !ok; v += 1) ok = String(v).startsWith(digit);
  leadingCache.set(key, ok);
  return ok;
}

function stepOk(value: number, spec: FieldSpec): boolean {
  return value >= 1 && value <= spec.max;
}

function stepped(start: number, end: number, step: number): Set<number> {
  const out = new Set<number>();
  for (let v = start; v <= end; v += step) out.add(v);
  return out;
}

// Values a *complete* term covers, or null if the term is not complete.
export function expand(term: string, spec: FieldSpec): Set<number> | null {
  if (term === '*') return stepped(spec.min, spec.max, 1);

  let m = /^\*\/([0-9]+)$/.exec(term);
  if (m && stepOk(Number(m[1]), spec)) return stepped(spec.min, spec.max, Number(m[1]));
  m = /^([0-9]+)$/.exec(term);
  if (m && spec.min <= Number(m[1]) && Number(m[1]) <= spec.max) return new Set([Number(m[1])]);
  m = /^([0-9]+)\/([0-9]+)$/.exec(term);
  if (m && stepOk(Number(m[2]), spec) && spec.min <= Number(m[1]) && Number(m[1]) <= spec.max) {
    return stepped(Number(m[1]), spec.max, Number(m[2]));
  }
  m = /^([0-9]+)-([0-9]+)$/.exec(term);
  if (m && spec.min <= Number(m[1]) && Number(m[1]) <= Number(m[2]) && Number(m[2]) <= spec.max) {
    return stepped(Number(m[1]), Number(m[2]), 1);
  }
  m = /^([0-9]+)-([0-9]+)\/([0-9]+)$/.exec(term);
  if (
    m &&
    stepOk(Number(m[3]), spec) &&
    spec.min <= Number(m[1]) &&
    Number(m[1]) <= Number(m[2]) &&
    Number(m[2]) <= spec.max
  ) {
    return stepped(Number(m[1]), Number(m[2]), Number(m[3]));
  }
  return null;
}

// Lists must be strictly ascending and non-overlapping; a start above the running maximum
// is enough, since a term's values all lie within [start, field max].
export function startOk(term: string, threshold: number | null, spec: FieldSpec): boolean {
  if (threshold === null) return true;
  if (term.length > 0 && term[0] === '*') return false;
  if (term.length === 0) {
    for (let v = spec.min; v <= spec.max; v += 1) if (v > threshold) return true;
    return false;
  }
  const m = /^[0-9]+/.exec(term);
  if (m === null) return true;
  const digits = m[0];
  if (digits.length === 2 || m[0].length < term.length) return Number(digits) > threshold;
  for (let v = spec.min; v <= spec.max; v += 1) {
    if (String(v).startsWith(digits) && v > threshold) return true;
  }
  return false;
}

// The trailing term answers two different questions: as a prefix only its start has to be
// placeable, as a field ending its whole expansion has to clear the running maximum.
export function listOk(s: string, spec: FieldSpec, trailingComplete: boolean): boolean {
  const parts = s.split(',');
  let threshold: number | null = null;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const values = expand(parts[i] as string, spec);
    if (values === null) return false;
    let lo = Infinity;
    let hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (threshold !== null && lo <= threshold) return false;
    threshold = hi;
  }
  const trail = parts[parts.length - 1] as string;
  if (!trailingComplete) return startOk(trail, threshold, spec);
  const values = expand(trail, spec);
  if (values === null) return false;
  if (threshold === null) return true;
  for (const v of values) if (v <= threshold) return false;
  return true;
}

const fieldStateCache = new Map<string, FieldState>();
const completionCache = new Map<string, number>();

// Bounded like the Python automaton's lru_cache: the breadth-first search probes far more
// partial fields than the reachable state space holds.
function cacheSet<K, V>(cache: Map<K, V>, key: K, value: V, max: number): void {
  if (cache.size >= max) cache.clear();
  cache.set(key, value);
}

// Fewest characters to append to reach a complete valid field, breadth first so the usual
// answers (0 or 1) are cheap. Only still-valid prefixes are carried forward: a failed parse
// stays failed however it grows.
export function fieldCompletionLen(text: string, spec: FieldSpec, maxExtra = 3): number {
  const key = `${spec.name}\u0000${text}\u0000${maxExtra}`;
  const hit = completionCache.get(key);
  if (hit !== undefined) return hit;

  let result = maxExtra + 1;
  if (fieldState(text, spec).canEnd) {
    result = 0;
  } else {
    let frontier = [''];
    for (let depth = 1; depth <= maxExtra; depth += 1) {
      const next: string[] = [];
      let found = false;
      for (const prefix of frontier) {
        for (const c of ALPHABET_CHARS) {
          const ext = prefix + c;
          const state = fieldState(text + ext, spec);
          if (state.canEnd) {
            found = true;
            break;
          }
          if (state.isPrefix) next.push(ext);
        }
        if (found) break;
      }
      if (found) {
        result = depth;
        break;
      }
      frontier = next;
    }
  }
  cacheSet(completionCache, key, result, 65536);
  return result;
}

export function fieldState(s: string, spec: FieldSpec): FieldState {
  const key = `${spec.name}\u0000${s}`;
  const hit = fieldStateCache.get(key);
  if (hit !== undefined) return hit;
  const result = fieldStateUncached(s, spec);
  cacheSet(fieldStateCache, key, result, 262144);
  return result;
}

function fieldStateUncached(s: string, spec: FieldSpec): FieldState {
  let state = 'TERM_START';
  let lo = 0;
  let one = 0;
  let i = 0;

  while (i < s.length) {
    const c = s[i] as string;
    if (state === 'TERM_START') {
      if (c === '*') {
        state = 'AFTER_STAR';
      } else if (isDigit(c) && leadingOk(c, spec.min, spec.max)) {
        one = Number(c);
        state = 'V1';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'AFTER_STAR') {
      if (c === '/') {
        state = 'BEFORE_STEP';
      } else if (c === ',') {
        state = 'TERM_START';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'V1') {
      if (isDigit(c)) {
        const v = one * 10 + Number(c);
        if (!(spec.min <= v && v <= spec.max)) return { isPrefix: false, canEnd: false };
        one = v;
        state = 'V2';
      } else if (c === '-' || c === '/' || c === ',') {
        if (!(spec.min <= one && one <= spec.max)) return { isPrefix: false, canEnd: false };
        if (c === '-') {
          if (one >= spec.max) return { isPrefix: false, canEnd: false };
          lo = one;
          state = 'R_LO';
        } else if (c === '/') {
          state = 'BEFORE_STEP';
        } else {
          state = 'TERM_START';
        }
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'V2') {
      if (c === '-') {
        if (!(spec.min <= one && one < spec.max)) return { isPrefix: false, canEnd: false };
        lo = one;
        state = 'R_LO';
      } else if (c === '/') {
        state = 'BEFORE_STEP';
      } else if (c === ',') {
        state = 'TERM_START';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'R_LO') {
      if (isDigit(c) && leadingOk(c, lo + 1, spec.max)) {
        one = Number(c);
        state = 'R_HI1';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'R_HI1') {
      if (isDigit(c)) {
        const v = one * 10 + Number(c);
        if (!(lo < v && v <= spec.max)) return { isPrefix: false, canEnd: false };
        one = v;
        state = 'R_HI2';
      } else if (c === '/' || c === ',') {
        if (one <= lo) return { isPrefix: false, canEnd: false };
        state = c === '/' ? 'BEFORE_STEP' : 'TERM_START';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'R_HI2') {
      if (c === '/' || c === ',') {
        state = c === '/' ? 'BEFORE_STEP' : 'TERM_START';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'BEFORE_STEP') {
      if (isDigit(c) && stepOk(Number(c), spec)) {
        one = Number(c);
        state = 'ST1';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'ST1') {
      if (isDigit(c)) {
        if (!stepOk(one * 10 + Number(c), spec)) return { isPrefix: false, canEnd: false };
        one = one * 10 + Number(c);
        state = 'ST2';
      } else if (c === ',') {
        state = 'TERM_START';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    } else if (state === 'ST2') {
      if (c === ',') {
        state = 'TERM_START';
      } else {
        return { isPrefix: false, canEnd: false };
      }
    }
    i += 1;
  }

  let syntactic: [boolean, boolean];
  if (state === 'TERM_START') syntactic = [s.length > 0, false];
  else if (state === 'V1') syntactic = [true, spec.min <= one && one <= spec.max];
  else if (state === 'R_LO') syntactic = [true, false];
  else if (state === 'R_HI1') syntactic = [true, one > lo];
  else if (state === 'BEFORE_STEP') syntactic = [true, false];
  else syntactic = [true, true];

  if (!syntactic[0]) return { isPrefix: false, canEnd: false };
  return {
    isPrefix: listOk(s, spec, false),
    canEnd: syntactic[1] && listOk(s, spec, true),
  };
}

export class CronAutomaton {
  readonly grammar: Grammar;
  private readonly allowedCache = new Map<string, ReadonlySet<string>>();

  constructor(grammar: Grammar) {
    this.grammar = grammar;
  }

  start(): State {
    return { field: 0, text: '', total: 0 };
  }

  // Length of the shortest completion that exists: a field hanging on a comma must open
  // *and* close its next term, which can take two digits. Gating allowed() on this keeps
  // completionLen(state) <= remaining budget at every reachable state.
  completionLen(state: { field: number; text: string }): number {
    const last = this.grammar.fields.length - 1;
    const spec = this.grammar.fields[state.field] as FieldSpec;
    return fieldCompletionLen(state.text, spec) + 2 * (last - state.field);
  }

  isTerminal(state: State): boolean {
    return state.field >= this.grammar.fields.length;
  }

  allowed(state: State): ReadonlySet<string> {
    if (this.isTerminal(state)) return new Set<string>();
    const key = `${state.field}\u0000${state.text}\u0000${state.total}`;
    const hit = this.allowedCache.get(key);
    if (hit !== undefined) return hit;

    const last = this.grammar.fields.length - 1;
    const spec = this.grammar.fields[state.field] as FieldSpec;
    const budget = this.grammar.maxLength - state.total;
    const out = new Set<string>();
    for (const c of this.grammar.alphabet) {
      if (c === ' ') continue;
      const text = state.text + c;
      const completes = this.completionLen({ field: state.field, text });
      if (fieldState(text, spec).isPrefix && 1 + completes <= budget) out.add(c);
    }
    if (fieldState(state.text, spec).canEnd && this.completionLen(state) <= budget) {
      out.add(state.field < last ? ' ' : EOS);
    }
    this.allowedCache.set(key, out);
    return out;
  }

  advance(state: State, ch: string): State | null {
    if (this.isTerminal(state) || !this.allowed(state).has(ch)) return null;
    if (ch === EOS || ch === ' ') return { field: state.field + 1, text: '', total: state.total + 1 };
    return { field: state.field, text: state.text + ch, total: state.total + 1 };
  }

  isComplete(state: State): boolean {
    if (this.isTerminal(state)) return true;
    if (state.field !== this.grammar.fields.length - 1) return false;
    return fieldState(state.text, this.grammar.fields[state.field] as FieldSpec).canEnd;
  }

  walk(text: string): State | null {
    let state: State | null = this.start();
    for (const ch of text) {
      if (state === null) return null;
      state = this.advance(state, ch);
    }
    return state;
  }
}

let defaultInstance: CronAutomaton | undefined;

export function defaultAutomaton(): CronAutomaton {
  defaultInstance ??= new CronAutomaton(defaultGrammar());
  return defaultInstance;
}

export function isWellFormed(text: string, automaton: CronAutomaton = defaultAutomaton()): boolean {
  const state = automaton.walk(text);
  return state !== null && automaton.isComplete(state);
}

export function parseCron(text: string): string[] {
  const parts = text.split(' ');
  if (parts.length !== 5) throw new Error(`expected 5 fields, got ${parts.length}: ${text}`);
  return parts;
}

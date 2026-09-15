import { describe, expect, it } from 'vitest';
import conformance from '../grammar/conformance.json';
import {
  EOS,
  defaultAutomaton,
  fieldState,
  isWellFormed,
  listOk,
  parseCron,
  startOk,
} from '../src/automaton.js';
import { defaultGrammar } from '../src/grammar.js';

interface StateCase {
  field: number;
  text: string;
  total: number;
  allowed: string;
  canEnd: boolean;
  completionLen: number;
}

interface ExpressionCase {
  text: string;
  wellFormed: boolean;
}

const auto = defaultAutomaton();
const grammar = defaultGrammar();
const states = conformance.states as StateCase[];
const expressions = conformance.expressions as ExpressionCase[];

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// total comes from the fixture: it is the budget the gate was evaluated at, not text.length.
function stateOf(c: StateCase) {
  return { field: c.field, text: c.text, total: c.total };
}

function allowedChars(state: { field: number; text: string; total: number }): string {
  return [...auto.allowed(state)]
    .filter((c) => c !== EOS)
    .sort()
    .join('');
}

describe('conformance', () => {
  it('loads the grammar the automaton was generated from', () => {
    expect(grammar.alphabet).toBe(conformance.alphabet);
    expect(grammar.maxLength).toBe(conformance.maxLength);
    expect(grammar.fields.map((f) => f.name)).toEqual(['minute', 'hour', 'dom', 'month', 'dow']);
  });

  it(`matches allowed() and canEnd for all ${states.length} states`, () => {
    // Exact counts move whenever the grammar tightens; the point is broad coverage.
    expect(states.length).toBeGreaterThan(12000);
    const tight = states.filter((c) => c.total >= grammar.maxLength - 5);
    expect(tight.length).toBeGreaterThan(9000);
    const mismatches: string[] = [];
    for (const c of states) {
      const state = stateOf(c);
      const got = allowedChars(state);
      const canEnd = auto.allowed(state).has(EOS);
      if (got !== c.allowed || canEnd !== c.canEnd) {
        mismatches.push(
          `field=${c.field} text=${JSON.stringify(c.text)} total=${c.total} allowed=${JSON.stringify(got)} (want ${JSON.stringify(c.allowed)}) canEnd=${canEnd} (want ${c.canEnd})`,
        );
      }
    }
    expect(mismatches.slice(0, 5)).toEqual([]);
    expect(mismatches).toHaveLength(0);
  });

  it(`matches completionLen for all ${states.length} states`, () => {
    // A function of (field, text) alone, so every budget variant of a base state shares it.
    const mismatches = new Set<string>();
    for (const c of states) {
      const got = auto.completionLen({ field: c.field, text: c.text });
      if (got !== c.completionLen) {
        mismatches.add(
          `field=${c.field} text=${JSON.stringify(c.text)} got=${got} want=${c.completionLen}`,
        );
      }
    }
    expect([...mismatches].slice(0, 5)).toEqual([]);
  });

  it(`matches isWellFormed for all ${expressions.length} expressions`, () => {
    expect(expressions.length).toBe(600);
    const mismatches = expressions.filter((e) => isWellFormed(e.text, auto) !== e.wellFormed);
    expect(mismatches.slice(0, 5)).toEqual([]);
    expect(mismatches).toHaveLength(0);
  });
});

describe('fieldState', () => {
  const minute = grammar.fields[0]!;

  it('accepts the reaches of the grammar and rejects its near misses', () => {
    for (const text of ['*', '*/5', '0', '59', '1-5', '1-5/2', '0,30', '0,30/2']) {
      expect(fieldState(text, minute), text).toEqual({ isPrefix: true, canEnd: true });
    }
    // Growable but not finishable: a reversed range, a dangling comma, a repeated value.
    for (const text of ['1-', '5-2', '1,', '3,3', '1,2/', '10/', '*/2,']) {
      expect(fieldState(text, minute), text).toEqual({ isPrefix: true, canEnd: false });
    }
    for (const text of ['', '60', '*/0', '*/60', '1,,2', '*,5', '-1', '/5', 'a']) {
      expect(fieldState(text, minute), text).toEqual({ isPrefix: false, canEnd: false });
    }
  });

  it('asks two different questions of the trailing term', () => {
    const text = '0-10,5';
    expect(listOk(text, minute, false)).toBe(true);
    expect(listOk(text, minute, true)).toBe(false);
    expect(startOk('5', 10, minute)).toBe(true);
    expect(startOk('11', 10, minute)).toBe(true);
    expect(startOk('10', 10, minute)).toBe(false);
    expect(startOk('9', 50, minute)).toBe(false);
    expect(startOk('', 58, minute)).toBe(true);
    expect(startOk('', 59, minute)).toBe(false);
    expect(startOk('*', null, minute)).toBe(true);
    expect(startOk('*', 10, minute)).toBe(false);
  });
});

describe('exploration', () => {
  it('never dead ends from any reachable state', () => {
    const rng = mulberry32(7);
    const seen = new Set<string>();
    const frontier = [auto.start()];
    for (let i = 0; i < 4000 && frontier.length > 0; i += 1) {
      const state = frontier.splice(Math.floor(rng() * frontier.length), 1)[0]!;
      const key = `${state.field}\u0000${state.text}\u0000${state.total}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const allowed = auto.allowed(state);
      expect(allowed.size, `dead end at ${key}`).toBeGreaterThan(0);
      // The invariant that makes that non-emptiness provable, checked on every reached state.
      expect(state.total + auto.completionLen(state)).toBeLessThanOrEqual(grammar.maxLength);
      for (const ch of allowed) {
        const next = auto.advance(state, ch);
        expect(next).not.toBeNull();
        // EOS is the end of the decoded string, so the state past it has no continuation.
        if (ch !== EOS && next !== null && frontier.length < 20000) frontier.push(next);
      }
    }
    expect(seen.size).toBeGreaterThan(500);
  });

  it('always finishes a random walk at a complete expression within the length cap', () => {
    const rng = mulberry32(11);
    for (let walk = 0; walk < 2000; walk += 1) {
      let state = auto.start();
      const chars: string[] = [];
      for (let step = 0; step < grammar.maxLength + 2; step += 1) {
        const allowed = [...auto.allowed(state)].sort();
        expect(allowed.length).toBeGreaterThan(0);
        const ch = allowed[Math.floor(rng() * allowed.length)]!;
        if (ch === EOS) break;
        chars.push(ch);
        const next = auto.advance(state, ch);
        expect(next).not.toBeNull();
        state = next!;
      }
      const text = chars.join('');
      expect(text.length).toBeLessThanOrEqual(grammar.maxLength);
      expect(auto.isComplete(state), `walk did not finish: ${JSON.stringify(text)}`).toBe(true);
      expect(isWellFormed(text, auto), text).toBe(true);
      expect(parseCron(text)).toHaveLength(5);
    }
  });

  it('never lets a walk past the length cap', () => {
    const rng = mulberry32(3);
    let deepest = 0;
    let deadEnds = 0;
    for (let walk = 0; walk < 500; walk += 1) {
      let state = auto.start();
      let text = '';
      for (;;) {
        const allowed = [...auto.allowed(state)];
        if (allowed.length === 0) {
          deadEnds += 1;
          break;
        }
        // Never take the separator early, so the budget is what forces the field to close.
        const chars = allowed.filter((c) => c !== EOS && c !== ' ');
        const ch =
          chars.length > 0
            ? chars[Math.floor(rng() * chars.length)]!
            : allowed.includes(' ')
              ? ' '
              : EOS;
        if (ch === EOS) break;
        text += ch;
        expect(text.length).toBeLessThanOrEqual(grammar.maxLength);
        state = auto.advance(state, ch)!;
      }
      deepest = Math.max(deepest, text.length);
      if (auto.isComplete(state)) expect(isWellFormed(text, auto), text).toBe(true);
    }
    // The cap only means anything if walks actually get near it.
    expect(deepest).toBeGreaterThan(40);
    expect(deadEnds).toBe(0);
  });

  it('prunes the comma that used to dead end a packed field', () => {
    // The comma needs 7 characters of budget (a two-digit term) and only 6 are left.
    const before = { field: 2, text: '5/7', total: 58 };
    expect(fieldState(before.text, grammar.fields[2]!).canEnd).toBe(true);
    expect(auto.completionLen(before)).toBe(4);
    expect(auto.allowed(before).has(',')).toBe(false);
    expect([...auto.allowed(before)]).toEqual([' ']);

    // The stuck state still exists as a value but is unreachable: only that comma led into it.
    const after = { field: 2, text: '5/7,', total: 59 };
    expect(fieldState(after.text, grammar.fields[2]!).canEnd).toBe(false);
    expect(auto.completionLen(after)).toBe(6);
    expect(auto.allowed(after).size).toBe(0);
    expect(grammar.maxLength - after.total).toBe(5);
  });
});

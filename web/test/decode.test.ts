import { describe, expect, it } from 'vitest';
import {
  CronAutomaton,
  EOS,
  defaultAutomaton,
  isWellFormed,
  type State,
} from '../src/automaton.js';
import {
  allowedIds,
  argmax,
  decode,
  maskLogits,
  newStats,
  type LogitsFn,
} from '../src/decode.js';
import {
  BOS,
  EOS_TOKEN_ID,
  PROMPT_SUFFIX,
  VOCAB_SIZE,
  decodeIds,
  encodePrompt,
} from '../src/tokenizer.js';
import { defaultGrammar } from '../src/grammar.js';

const auto = defaultAutomaton();
const grammar = defaultGrammar();

function spike(id: number, height = 10): Float32Array {
  const logits = new Float32Array(VOCAB_SIZE).fill(-5);
  logits[id] = height;
  return logits;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// A fake that reads off one fixed expression, one character per call.
function follow(target: string): LogitsFn {
  let position = 0;
  return async () => {
    const at = position;
    position += 1;
    return spike(at < target.length ? target.charCodeAt(at) : EOS_TOKEN_ID);
  };
}

// 'Z' is not in the alphabet, so this fake only ever asks for illegal tokens.
const hostile: LogitsFn = async () => {
  const logits = spike('Z'.charCodeAt(0), 100);
  logits['9'.charCodeAt(0)] = 50;
  return logits;
};

const uniform: LogitsFn = async () => new Float32Array(VOCAB_SIZE).fill(1);

describe('tokenizer', () => {
  it('encodes BOS plus the UTF-8 of "text => "', () => {
    const ids = encodePrompt('every weekday at 9am');
    expect(ids[0]).toBe(BOS);
    expect(decodeIds(ids.slice(1))).toBe(`every weekday at 9am${PROMPT_SUFFIX}`);
  });

  it('round trips thrown bytes and stops at EOS', () => {
    const cron = '*/15 0-23/2 * JAN *';
    const ids = [...cron].map((c) => c.charCodeAt(0));
    expect(decodeIds([...ids, EOS_TOKEN_ID])).toBe(cron);
    expect(decodeIds([...ids, EOS_TOKEN_ID, 65])).toBe(cron);
  });
});

describe('masking', () => {
  it('changes the argmax when the favourite token is illegal', () => {
    const state = auto.start();
    const raw = spike('Z'.charCodeAt(0), 100);
    expect(argmax(raw)).toBe('Z'.charCodeAt(0));
    const allowed = allowedIds(auto, state);
    expect(allowed.has('Z'.charCodeAt(0))).toBe(false);
    const masked = maskLogits(raw, allowed);
    expect(allowed.has(argmax(masked))).toBe(true);
    expect(masked.length).toBe(VOCAB_SIZE);
  });

  it('maps the grammar sentinel to the EOS token and every other char to its byte', () => {
    const first = auto.advance(auto.start(), '*')!;
    expect(auto.allowed(first).has(' ')).toBe(true);
    expect(auto.allowed(first).has(EOS)).toBe(false);
    expect(allowedIds(auto, first).has(' '.charCodeAt(0))).toBe(true);

    let last = auto.start();
    while (!auto.isComplete(last)) {
      const next = [...auto.allowed(last)].find((c) => c !== EOS)!;
      last = auto.advance(last, next)!;
    }
    expect(auto.allowed(last).has(EOS)).toBe(true);
    expect(allowedIds(auto, last).has(EOS_TOKEN_ID)).toBe(true);
  });
});

describe('constrained decoding', () => {
  it('follows a fake that prefers one fixed expression', async () => {
    const result = await decode('whenever', follow('0 9 * * 1-5'));
    expect(result.text).toBe('0 9 * * 1-5');
    expect(result.truncated).toBe(false);
    expect(isWellFormed(result.text)).toBe(true);
    expect(result.steps).toBe('0 9 * * 1-5'.length);
    expect(result.tokenMillis).toHaveLength(result.steps + 1);
  });

  it('still yields a well formed expression when every step prefers an illegal token', async () => {
    const result = await decode('whenever', hostile);
    expect(isWellFormed(result.text)).toBe(true);
    expect(result.truncated).toBe(false);
    expect(result.text).toMatch(/^[0-9*,/-]+( [0-9*,/-]+){4}$/);
  });

  it('reaches EOS for any logits function within the length cap', async () => {
    const rng = mulberry32(5);
    const random: LogitsFn = async () => {
      const logits = new Float32Array(VOCAB_SIZE);
      for (let i = 0; i < VOCAB_SIZE; i += 1) logits[i] = rng() * 4 - 2;
      return logits;
    };
    for (const fake of [hostile, uniform, random]) {
      for (let i = 0; i < 40; i += 1) {
        const result = await decode('every weekday at 9am', fake);
        expect(result.truncated).toBe(false);
        expect(isWellFormed(result.text), result.text).toBe(true);
        expect(result.text.length).toBeLessThanOrEqual(grammar.maxLength);
      }
    }
  });

  it('counts truncations when the budget is cut short', async () => {
    const stats = newStats();
    const result = await decode('x', hostile, { maxNew: 3, stats });
    expect(result.truncated).toBe(true);
    expect(stats.truncated).toBe(1);
    expect(stats.decodes).toBe(1);
    expect(result.steps).toBe(3);
    expect(isWellFormed(result.text)).toBe(false);
  });

  it('steers away from a prefix the automaton will not let it finish', async () => {
    // The comma cannot be closed inside the remaining budget, so the automaton prunes it and
    // the fake is forced onto a different, completable last field.
    const packed = '7/42,50-52,54-55/20,57/16,58,59-59/36 8,19-22,23-23/11 5/7,';
    const result = await decode('x', follow(packed));
    expect(result.text).not.toBe(packed);
    expect(result.truncated).toBe(false);
    expect(isWellFormed(result.text), result.text).toBe(true);
  });

  it('throws when the automaton offers no move at all', async () => {
    class StuckAutomaton extends CronAutomaton {
      override allowed(): ReadonlySet<string> {
        return new Set<string>();
      }
    }
    await expect(decode('x', hostile, { automaton: new StuckAutomaton(grammar) })).rejects.toThrow(
      /allowed no move/,
    );
  });

  it('throws when the automaton refuses a character it just allowed', async () => {
    class LiarAutomaton extends CronAutomaton {
      override advance(): State | null {
        return null;
      }
    }
    await expect(decode('x', uniform, { automaton: new LiarAutomaton(grammar) })).rejects.toThrow(
      /rejected its own allowed character/,
    );
  });

  it('refuses a logits vector of the wrong width', async () => {
    const wrong: LogitsFn = async () => new Float32Array(10);
    await expect(decode('x', wrong)).rejects.toThrow(/logits width/);
  });

  it('hands the model the prompt plus everything generated so far', async () => {
    const target = '0 9 * * 1-5';
    const inner = follow(target);
    const seen: number[][] = [];
    const spy: LogitsFn = async (ids) => {
      seen.push([...ids]);
      return inner(ids, ids.length);
    };
    await decode('hello', spy);
    const prompt = encodePrompt('hello');
    expect(seen[0]).toEqual(prompt);
    expect(seen[1]!.slice(0, prompt.length)).toEqual(prompt);
    expect(seen[1]!.length).toBe(prompt.length + 1);
    // One call per emitted character plus the call that picks EOS.
    expect(seen).toHaveLength(target.length + 1);
    expect(seen.at(-1)!.length).toBe(prompt.length + target.length);
  });
});

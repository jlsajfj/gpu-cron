import raw from '../../grammar/cron-grammar.json';

export interface FieldSpec {
  readonly name: string;
  readonly min: number;
  readonly max: number;
}

export interface Grammar {
  readonly fields: readonly FieldSpec[];
  readonly alphabet: string;
  readonly maxLength: number;
}

interface RawGrammar {
  fields: { name: string; min: number; max: number; index: number }[];
  alphabet: string;
  maxLength: number;
}

export function parseGrammar(json: RawGrammar): Grammar {
  const fields = [...json.fields]
    .sort((a, b) => a.index - b.index)
    .map((f) => ({ name: f.name, min: f.min, max: f.max }));
  return { fields, alphabet: json.alphabet, maxLength: json.maxLength };
}

let cached: Grammar | undefined;

export function defaultGrammar(): Grammar {
  cached ??= parseGrammar(raw as RawGrammar);
  return cached;
}

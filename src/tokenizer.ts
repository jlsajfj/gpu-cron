export const PAD = 256;
export const BOS = 257;
export const EOS_TOKEN_ID = 258;
export const VOCAB_SIZE = 259;

export const PROMPT_SUFFIX = ' => ';

// Day and month names in the corpus are capitalized ~99.95% of the time (6,388 "Tuesday"
// against 0 "tuesday"), and the vocabulary is bytes, so the capitalized form is a different
// sequence the model would otherwise never see. Lowercasing here — rather than duplicating
// every pair — leaves exactly one form to learn. train/data.py lowercases identically.
export function encodePrompt(text: string): number[] {
  return [BOS, ...new TextEncoder().encode(text.toLowerCase() + PROMPT_SUFFIX)];
}

export function decodeIds(ids: readonly number[]): string {
  const bytes: number[] = [];
  for (const id of ids) {
    if (id === EOS_TOKEN_ID) break;
    if (id === PAD || id === BOS || id > 255) continue;
    bytes.push(id);
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

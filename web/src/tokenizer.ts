export const PAD = 256;
export const BOS = 257;
export const EOS_TOKEN_ID = 258;
export const VOCAB_SIZE = 259;

export const PROMPT_SUFFIX = ' => ';

// The model is trained on bytes, so the prompt is BOS followed by the UTF-8 of "text => ".
export function encodePrompt(text: string): number[] {
  return [BOS, ...new TextEncoder().encode(text + PROMPT_SUFFIX)];
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

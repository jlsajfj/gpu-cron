// Shipped to dist/index.d.ts. The surface is small enough to state by hand, and hand-written
// declarations are what keep the package one file with no internal type imports.

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
  adapter: string;
  params: number;
  bytes: number;
}

export declare class CronError extends Error {}

export declare function parse(text: string, options?: ParseOptions): Promise<CronMatch>;

/** Which path the first parse() took. Null until something has been parsed. */
export declare function backend(): Backend | null;

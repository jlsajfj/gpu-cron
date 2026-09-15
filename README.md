# human-cron

Natural language to cron, running entirely in your browser.

```
"every weekday morning at 9"   ->  0 9 * * 1-5
"2:22pm on the 5th of january" ->  22 14 5 1 *
"every second hour"            ->  0 */2 * * *
```

A **45,376 parameter** transformer, trained from scratch on a generated dataset of
(English phrasing, cron expression) pairs, plus a constrained decoder that masks the
model's logits against a grammar automaton so every emitted string is valid cron *by
construction*. Weights are inlined in the page: no server, no API call, no
post-processing.

```js
import { isAvailable, parse } from 'human-cron';

if (await isAvailable()) {
  await parse('every weekday at 9am');
  // { expression: '0 9 * * 1-5', next: ['2026-01-01T09:00:00.000Z', ...] }
}
```

> **Requires WebGPU.** This is a browser package. There is no CPU fallback and none is
> planned — see [Why there is only one backend](#why-there-is-only-one-backend).
> `isAvailable()` resolves `false` in Node rather than throwing, so it is always safe to
> call.

The whole package is **one 89 KB file** (50 KB brotli), weights included.

## API

Two functions. `isAvailable()` asks whether this environment can run the model; `parse()`
does the work.

### `isAvailable(): Promise<boolean>`

Whether this environment can run the model. Resolves `false` instead of throwing, so it is
safe to call anywhere — including in Node, where it is always `false`.

The first call uploads the weights to the GPU, which is the expensive part (tens of
milliseconds). **That work is shared with `parse()`**, so calling `isAvailable()` first
makes the first `parse()` fast rather than doing the upload twice. Repeat calls are free.

```js
if (!(await isAvailable())) {
  showPlainCronInput();   // fall back to a text field
  return;
}
```

### `parse(text, options?): Promise<CronMatch>`

```ts
interface CronMatch {
  expression: string;   // always syntactically valid cron
  next: string[];       // upcoming fire times, ISO-8601
}
```

`options.count` sets how many fire times to return (default 5).

The `expression` is **always syntactically valid cron** — the decoder masks the model's
logits against a grammar automaton, so an invalid string is unreachable rather than merely
unlikely. It is **not** guaranteed to be the schedule you meant. This is a 45k-parameter
model with no way to signal that it did not understand you: nonsense input yields a
confident, valid, meaningless expression. Show the user what you parsed.

Throws `CronError` on failure.

### `unavailable(): CronError | null`

The error explaining why the model is unavailable, or `null` if it is available. This is
the **same instance** `parse()` would throw, handed to you without having to call `parse()`
and catch — so a tooltip can never disagree with an error toast.

Latched: once the runtime fails it stays failed, so it is safe to read during render.

```js
if (!(await isAvailable())) {
  const err = unavailable();
  input.disabled = true;
  input.title = err instanceof NoWebGpuError ? 'GPU not available' : 'Unavailable';
}
```

### Errors

Branch with `instanceof`, not on strings. Everything descends from `CronError`.

| class | meaning | recoverable? |
|---|---|---|
| `NoWebGpuError` | no WebGPU adapter — this is what Node gets | no |
| `NoModelError` | the build has no weights inlined; a packaging bug | no |
| `InferenceFailedError` | WebGPU present, but the pipeline would not load or a run threw | no |
| `InputTooLongError` | the prompt does not fit alongside the answer | **yes** — shorten the input |
| `UngrammaticalError` | the decoder could not close a legal expression | no — this should be unreachable; please report it |

The first three latch and are what `unavailable()` returns: disable your UI on those.
`InputTooLongError` is per-input, so show an inline message and leave the field enabled.

`UngrammaticalError` is an internal invariant check rather than a user-facing condition.
The automaton cannot enter a state it is unable to close before the cap, and it has not
fired across ~21,000 eval examples or any adversarial probe. If you see it, it is a bug
here.

**Nothing is truncated on your behalf.** An over-long prompt is a hard error, because a
silently clipped prompt decodes to a schedule nobody asked for.

### `backend(): Backend | null`

Diagnostics. `{ adapter, params, bytes }` once the model has loaded, `null` before. Useful
for showing which GPU answered.

## Results

<!--RESULTS_TABLE-->
| model | split | n | semantic | exact | valid | fires |
|---|---|---:|---:|---:|---:|---:|
| 45k | test (unseen expression) | 8,998 | **86.5%** | 84.0% | 100.0% | 100.0% |
| 45k | holdout (unseen phrasing, seen expression) | 11,795 | **82.8%** | 79.8% | 100.0% | 100.0% |

Same weights, mask on vs. off, on the same held-out examples:

| model | decoding | valid cron | semantic | exact |
|---|---|---:|---:|---:|
| 45k | constrained | 100.0% | 94.9% | 94.9% |
| 45k | unconstrained | 100.0% | 94.7% | 94.7% |

n = 1000 held-out examples. The mask does not just make the output valid — on both models it is also the more accurate of the two, and the constrained column is valid cron by construction rather than by measurement.
<!--/RESULTS_TABLE-->

## Why constrained decoding

The usual way to keep a language model's output well-formed is to repair the string
afterwards with a regex. This project does not do that.

Note what the results table says: on in-distribution held-out prompts the *unconstrained*
model is already 100% valid, so the mask is not earning its place by fixing a failure rate.
It earns it by changing the kind of claim you can make. "Valid 100% of the time on the
sample we measured" and "cannot emit an invalid string" are different statements, and only
the second one survives contact with input nobody anticipated.

Twenty deliberately hostile inputs — gibberish, an empty string, `DROP TABLE users; --`,
180 random characters, emoji — produced **20 valid cron expressions out of 20**, because
producing anything else is unreachable.

At every step an automaton over the cron grammar reports which characters keep the output
on a path to a complete expression, every other logit is set to `-inf`, and the argmax is
taken over what remains. The guarantee is structural, not empirical: a decoded string is
well-formed because each character was chosen from the set of characters that preserve
well-formedness.

`completionLen()` keeps the decoder from wandering into a corner it cannot close before
the length cap, which is what makes "there is always a legal move" an invariant rather
than a hope.

The automaton encodes semantics, not just shape: `*/0` is illegal, list terms must be
strictly ascending so `3,2` and `1,1` are untypeable, and a range's high end must strictly
exceed its low end so `6-6` cannot be spelled.

It exists twice — Python (`eval/cron_automaton.py`, used by training and eval) and
TypeScript (`src/automaton.ts`, used by the browser). Two implementations of one grammar
drift, so `grammar/conformance.json` freezes **12,499 reachable automaton states** and 600
accept/reject cases generated from the Python side, and the browser suite asserts against
the same file.

## The dataset is generated, not labelled

There is no corpus of real cron requests, so both halves are synthesised. The recurring
lesson of this project is that **almost every quality bug was a generator bug** — the
generator narrowing its own output space, and the model getting blamed for the gap.

**Stage 1 — expressions (`data/build-canonical.mjs`).** Cron expressions sampled from a
weighted distribution of 24 shapes people actually write. `cronstrue` is the inverse
function (cron to canonical English); `cron-parser` validates, so `0 0 30 2 *` cannot
enter the dataset.

**Stage 2 — phrasings (`data/paraphrase.mjs`).** Each canonical string is expanded into
8-40 surface phrasings by an LLM (`gpt-4.1-mini`), varying length, register and clock
style. Phrasings are filtered for timezone and relative-date leakage and de-duplicated
globally.

**Stage 2b — canonical echo (`data/assemble.py`).** cronstrue's own string is added back
verbatim. Not padding: the LLM is asked for *diverse* phrasings and systematically avoids
the plainest reading, so `*/15 * * * *` had forty paraphrases and not one of them was
"every 15 minutes".

**Stage 2c — surface-form coverage (`data/augment.mjs`).** The same failure one level
down. The LLM is uneven about how it *spells* things, so the model learned spellings
rather than values:

| what was missing | consequence |
|---|---|
| the word `second` — **0 rows in 122,711** (the phrasing filter banned `/seconds?/` to exclude sub-minute intervals, and that also ate the ordinal and the step) | `second of the month` and `first of the month` decoded identically |
| 53 of 60 minute values — `TIDY_MINUTES = [0,15,30,45]` was 99.2% of rows | `3:21pm` decoded to minute **15**, snapping to the mode |
| February, April, May, August, October as a *sole* month value | `8am on august 3rd` had no shape to copy |
| ordinals against the minute field | `the 12th minute` decoded as `*/12` |

This stage walks the grid directly — every value, in every spelling (`2`, `2nd`, `second`,
`two`), in every field it can legally occupy — rather than hoping the LLM covers it. It is
templated and seeded, so it costs nothing and regenerates byte-identically.

**Stage 3 — splits (`data/assemble.py`).** Held out at two levels:

| split | what it measures |
|---|---|
| `test` | the expression *and* its canonical English were never trained on |
| `holdout` | one phrasing per training expression — isolates "new way of saying a known schedule" from "new schedule" |

The finished dataset is **167,967 pairs over 12,270 distinct expressions**: 141,645 train,
5,529 val, 8,998 test, 11,795 holdout.

**A structural caveat worth stating plainly.** Both splits are held out from the *same
generator*, so a generator that emits only four minute values produces validation data
containing only those four, and reports an excellent score for a model that fails on
`:21`. Held-out data from a broken generator validates the breakage. Every bug in the
table above was invisible to `val_loss` and surfaced only through hand-written probes and
per-bucket eval slices.

### Cost

Stage 2 is the only step that calls an API: ~1,470 batch calls, ~$2, for 132k phrasings.
Stage 2c is templated and free. `gpt-5-mini` was the first choice and was 6x slower for no
quality gain — it spent 2,000-3,000 hidden reasoning tokens per call on a task with no use
for reasoning.

## Ambiguity policies

Ambiguity is the main quality risk, and the model can only be as consistent as the
dataset. Every policy is applied uniformly at generation time. These are *choices*, not
universal truths about cron.

| ambiguity | policy |
|---|---|
| "morning" / "afternoon" / "evening" / "night" | 09:00 / 15:00 / 18:00 / 21:00 |
| "noon", "midnight" | 12:00, 00:00 |
| "every 15 minutes between 9 and 5" | hour ranges are inclusive: `9-17` |
| naming a day but not a time | not generated — avoids inventing times the user never said |
| day-of-month *and* day-of-week both restricted | never generated: cron flips from AND to OR and no phrasing disambiguates it |
| Sunday | `0`; `7` is valid cron and never emitted |
| month and day names in output (`JAN`, `MON`) | valid cron, never emitted |
| `@daily`-style macros | unsupported; always 5 fields |
| timezones | out of scope; cron has no timezone field |
| "end of month" | not generated — needs `L`, which this dialect lacks |
| list ordering | strictly ascending and non-overlapping |
| "the second of the month" | ordinal: day-of-month `2` |
| "every second hour" | step: `0 */2 * * *`; identical to "every other hour" |
| "every 30 seconds" and any sub-minute interval | filtered from the dataset. The decoder still emits *some* expression (the grammar cannot decline), so this is a known wrong answer, not a refusal |
| which spelling a number uses | never meaningful alone: `2`, `2nd`, `second`, `two` are one value; context decides the field |

## How the model works

- **Vocabulary.** Bytes, plus PAD/BOS/EOS — 259 total. No tokenizer. Input is free English
  and the output alphabet is 16 characters, so one byte vocabulary covers both sides and
  the decoder only ever masks logits, never translates between tokenizers.
- **Format.** `BOS + "<english> => " + cron + EOS`, loss on the cron bytes and EOS only.
  The prompt is lowercased at every encoder and masked out of the loss.
- **Architecture.** Decoder-only: `d_model 32`, 3 layers, 2 heads, `d_ff 96`, `max_len
  192`. Pre-LN RMSNorm, learned absolute positions, GELU MLP, tied embeddings, causally
  masked `scaled_dot_product_attention`. Scaled residual init on the projections, without
  which the first few hundred steps are spent recovering a sane residual scale.
- **Batching.** Length-bucketed with a window shuffle — sorting by length alone groups one
  expression's near-identical paraphrases into a batch and collapses the loss to
  memorisation.
- **Quantization.** int8 per output row with a float32 scale per row. RMSNorm gains stay
  float32: a few hundred numbers that scale the whole residual stream.

## Why there is only one backend

The forward pass is ten WGSL compute shaders (`src/gpu.ts`). A browser without an adapter
gets a clear error rather than a second, slower implementation that can disagree with the
first. The arithmetic is the deliverable, so there is exactly one copy of it.

The grammar mask stays on the CPU: one readback per token (259 floats) is the price of
constrained decoding, and reimplementing the automaton in WGSL would not pay for itself.

Failure states are latched and classified rather than retried: `no-model`, `no-webgpu`,
`inference-failed`. The demo in `demo/` imports the package the way an npm user would, so
it cannot drift from the published API.

## Repository layout

```
data/      dataset generation (Node for cronstrue/cron-parser, Python for assembly)
grammar/   the cron dialect + the Python/JS agreement fixture
train/     model, training loop, and the unverified SmolLM2 GPU path
eval/      the Python automaton, constrained decoder, and eval harness
export/    checkpoint -> int8 weights + manifest
src/       the published package: automaton, decoder, WGSL runtime, inlined weights
demo/      the browser demo
tests/     Python property tests    test/  TypeScript unit + conformance tests
```

## Reproducing

```bash
make setup       # venv + CPU torch, npm ci
make data        # stage 1, the API-backed stage 2, then seeded stage 2c
make train-pico  # the shipped model, ~22 min on 16 CPU cores
make eval        # exact + semantic match on test and holdout (CKPT=runs/<run>/checkpoint.pt)
make web         # export int8 weights and bundle dist/ + demo/dist/
make demo        # build and serve the demo — the thing to run after a clone
make test        # automaton property tests + typecheck + unit tests
```

Everything in the tested path runs on CPU; the box this was developed on has 16 ARM
Neoverse-V2 cores, 61 GB RAM and no GPU. `--bf16` is about 1.6x faster than fp32 there.
`configs/smoke.json` trains in seconds so the suite can exercise the whole loop.

`train/finetune_smollm2.py` fine-tunes SmolLM2-135M with LoRA for a machine with a GPU. It
is **not** the default, nothing in the tested path depends on it, and it has never been
executed here — treat it as a starting point, not a verified path.

NixOS note: the manylinux torch wheel needs a real `libstdc++.so.6`, which nix's Python
does not put on the default search path. `bin/py` is a two-line wrapper that sets
`LD_LIBRARY_PATH`; every Python entry point goes through it.

## Limitations

**It is a 45k-parameter model trained on generated data.** It is not a replacement for a
well-tested cron library. Treat a wrong answer as likely on unusual phrasings.

**There is no way for it to say "I don't know."** The automaton always has a legal next
character, so nonsense input yields a confident, valid, meaningless expression — the empty
string decodes to `0 9 * */2 *`. Probability mass on legal characters is 0.99+ for
gibberish and real prompts alike, so there is no free confidence signal either. A rejection
path needs negatives in training, of which there are currently none.

**The grammar is narrower than cron.** Lists must be ascending and non-overlapping, so
`*/2,5` is rejected even though cron-parser accepts it. The guarantee runs one way: every
output is valid cron, but not every valid cron is reachable.

**Validity is syntactic; satisfiability is checked afterwards.** The automaton cannot know
that day 30 of February never fires. The eval harness verifies every expression with
cron-parser (`fires_pct`) rather than repairing it.

**English only.** `я хочу каждый вторник` and `¿cada martes a las nueve?` both decode to
something confident and wrong.

**Phrasings are LLM-generated and not individually verified.** Nothing re-derives the
schedule from each phrasing to confirm it still means the same thing. A wrong paraphrase
is a wrong training label. Sampling found them faithful; that is a spot check, not a proof.

**The splits share a little data.** Splitting is by expression, but the sampler can reach
the same cron string from two buckets, so a small fraction of val and test expressions also
appear in training. No (phrasing, cron) pair appears in two splits.

**Two bugs the verification caught.** Training labels were originally aligned with the
input rather than shifted by one, turning the task into copying the current token — both
losses went to ~0.0 while greedy decoding produced nothing but `* * * * *`. And the
decoder's completion-length estimate assumed a field dangling on a comma could be closed in
one character, which let the length budget strand it in an empty `allowed()`. Both were
invisible to loss curves and surfaced only because the decoder was exercised end to end.

## License

MIT.

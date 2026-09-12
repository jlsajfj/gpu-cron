# nl-cron

Natural language to cron, running entirely in your browser.

A ~25M parameter transformer trained from scratch on a generated dataset of
(English phrasing, cron expression) pairs, plus a constrained decoder that masks the
model's logits against a grammar automaton so every emitted string is valid cron by
construction. The demo runs the quantized model on WebGPU with the weights inlined in
the page — no server, no API call, no post-processing.

```
"every weekday morning at 9"  ->  0 9 * * 1-5
```

## Results

<!--RESULTS_TABLE-->

## Why constrained decoding

An unconstrained model of this size emits malformed cron regularly — `*/5 * * 1-5`,
`0 9 * * 8`, `0 25 * * *`, trailing junk. The usual fix is to repair the string
afterwards with a regex. This project does not do that: at every decoding step an
automaton over the cron grammar reports the characters that keep the output on a path
to a complete expression, every other logit is set to `-inf`, and the argmax is taken
over what remains.

The guarantee is structural, not empirical: a decoded string is well-formed because
each character was chosen from the set of characters that preserve well-formedness. The
automaton's `allowed()` is a lazily-expanded trie walk over the grammar's language;
`completionLen()` keeps the decoder from wandering into a corner it cannot close before
the length cap, which is what makes "there is always a legal move" a real invariant
rather than a hope.

The same automaton runs twice — Python (`eval/cron_automaton.py`, used by training and
eval) and TypeScript (`web/src/automaton.ts`, used by the browser). Two implementations
of one grammar drift, so `grammar/conformance.json` freezes 12,500 automaton states (2,500 base states replayed at several length budgets) and
600 accept/reject cases from the Python side and the browser test suite asserts against
the same file.

## The dataset is generated, not labelled

There is no corpus of real cron requests to train on, so both halves are synthesised.

**Stage 1 — expressions (`data/build-canonical.mjs`).** Cron expressions are sampled from
a weighted distribution of 24 shapes people actually write: every N minutes, every N
hours at minute M, daily at HH:MM, weekdays, weekends, weekly on a day, monthly on the
Nth, monthly on a range or list, business hours, hour ranges and steps, day-of-week
ranges/lists/steps, month ranges/lists, yearly, and offset steps like `5/20`. Quotas are
per *bucket* and measured in finished pairs, not distinct expressions, so the
high-cardinality buckets cannot crowd out `*/5 * * * *` — an earlier version of this
script sampled until it had N distinct strings and produced a dataset that was 26%
`hour-range-step` and 0.3% "daily at 9am".

`cronstrue` is the inverse function (cron to canonical English) and `cron-parser` is the
validator: every sample is expanded to real fire times and dropped if it never fires, so
`0 0 30 2 *` cannot enter the dataset.

**Stage 2 — phrasings (`data/paraphrase.mjs`).** Each canonical English string is
expanded into 8–40 surface phrasings by an LLM (`gpt-4.1-mini`), varying length,
register, and clock style. Buckets with few distinct expressions get more phrasings
each, so the finished pair distribution still tracks the intended weights. Phrasings are
filtered for timezone/relative-date leakage and de-duplicated globally; a phrasing that
maps to two different expressions is dropped rather than assigned an arbitrary winner.

**Stage 2b — canonical echo (`data/assemble.py`).** cronstrue's canonical string is itself a
valid phrasing, and it is added verbatim as a training pair for every expression. This is
not padding: the LLM is asked for *diverse* phrasings and systematically avoids the
plainest reading, so `*/15 * * * *` — one of the most common schedules there is — had
forty paraphrases and not one of them was "every 15 minutes". The model had never seen the
most obvious way to say the most common schedule. Adding the canonical form back costs
nothing and makes the plain register representable for all 10,797 expressions.

**Stage 3 — splits (`data/assemble.py`).** Splits hold out at two levels, and both are
reported:

| split | what it measures |
|---|---|
| `test` | the expression *and* its canonical English were never trained on |
| `holdout` | one phrasing per training expression, held back before training — isolates "new way of saying a known schedule" from "new schedule" |

The finished dataset is **144,483 pairs over 10,797 distinct expressions**: 122,711 train,
4,372 val, 7,608 test, 9,792 holdout.

### Cost

Generating the dataset costs real money and is not run by the test suite. Stage 2 is the
only step that calls an API.

| stage | calls | est. cost |
|---|---|---|
| stage 2 paraphrasing (gpt-4.1-mini) | ~1,470 batch calls, 132k phrasings | **~$2** |

Roughly 0.7M input and 1.1M output tokens at list pricing. `gpt-5-mini` was the first
choice and was 6x slower here for no quality gain — it spent 2,000-3,000 hidden reasoning
tokens per call, which paraphrasing has no use for.

## Ambiguity policies

Ambiguity is the main quality risk, and the model can only be as consistent as the
dataset. Every policy below is applied uniformly at generation time and enforced by the
grammar where the grammar can express it. These are *choices*, not universal truths
about cron.

| ambiguity | policy | why |
|---|---|---|
| "morning" / "afternoon" / "evening" / "night", no clock time | 09:00 / 15:00 / 18:00 / 21:00 | cron has no fuzzy times; a fixed anchor beats emitting nothing |
| "noon", "midnight" | 12:00, 00:00 | unambiguous |
| "every 15 minutes between 9 and 5" | hour ranges are inclusive: `9-17` | "between" is read as the closed business-hours range |
| "every morning" with no time | 09:00, same as "morning" | one anchor per period word, everywhere |
| Naming a day but not a time | not generated | avoids inventing times the user never said |
| Day-of-month *and* day-of-week both restricted | never generated | cron flips from AND to OR when both are restricted; no phrasing could disambiguate, so the dataset refuses to teach it |
| Sunday | `0`. `7` is valid cron and never emitted | keeps output canonical |
| Month and day names (`JAN`, `MON`) | valid cron, never emitted | same reason |
| `@daily`-style macros | not supported; always 5 fields | keeps the output space a single grammar |
| Timezones | out of scope entirely | cron has no timezone field; a phrasing mentioning one is filtered out of the dataset |
| "end of month" | not generated | needs `L`, which the target dialect does not have; the nearest expressible thing (the 28th) is not what the user meant |
| Leading zeros (`05 * * * *`) | valid, never emitted | canonical output |
| List ordering | terms must be strictly ascending and non-overlapping | narrows the language slightly (cron-parser also allows `*/2,5`) but makes "no duplicate values" checkable one character at a time, and matches how people write lists |
| Steps (`*/0`, `5-2`) | rejected by the grammar | not valid cron |

One policy is enforced by the automaton and one only by the dataset: the grammar rejects
syntactically-invalid and duplicate-value lists, but it cannot know that day 30 of
February never fires. The decoder's output is checked against `cron-parser` after the
fact as a *verification* (see "Limitations"), never as a repair.

## Repository layout

```
data/           dataset generation (Node for cronstrue/cron-parser, Python for assembly)
grammar/        the cron dialect + the Python/JS agreement fixture
train/          model, training loop, and the unverified SmolLM2 GPU path
eval/           the automaton, the constrained decoder, and the eval harness
tests/          Python property tests for the automaton
web/            the browser demo (ONNX Runtime Web, WebGPU)
export/         checkpoint -> ONNX + int8 quantization
```

## Reproducing

```bash
make setup        # venv + CPU torch + numpy, and npm install in web/
make data         # ~1 min for stage 1, then the API-backed stage 2
make train        # the default CPU run (configs/default.json)
make train-mini   # the browser-sized model
make eval         # exact + semantic match on test and holdout
make web          # export int8 ONNX and bundle the demo
make test         # automaton property tests + browser unit tests
```

### Hardware

Everything in the tested path runs on CPU. On the box this was developed on — 16 ARM
Neoverse-V2 cores, 61 GB RAM, no GPU — `configs/default.json` (~25M params) is the
overnight configuration and `configs/mini.json` (~3M params) is the browser model.
`configs/smoke.json` trains in seconds and exists so the test suite can run the whole
loop end to end.

`train/finetune_smollm2.py` fine-tunes SmolLM2-135M with LoRA for a machine that has a
GPU. It is **not** the default, **nothing in the tested path depends on it**, and it has
not been executed on the box (no GPU, and `transformers` is not installed) — treat it as
a starting point, not a verified path.

NixOS note: the manylinux torch wheel needs a real `libstdc++.so.6`, which nix's Python
does not put on the default search path. `bin/py` is a two-line wrapper that sets
`LD_LIBRARY_PATH`; every Python entry point goes through it.

## How the model works

- **Vocabulary.** Bytes, plus PAD/BOS/EOS. The input is free English and the output
  alphabet is 16 characters, so one byte vocabulary covers both sides and the
  constrained decoder only ever masks logits — it never translates between tokenizers.
- **Format.** `BOS + "<english> => " + cron + EOS`, with the loss taken on the cron
  bytes and EOS only. The prompt is masked out.
- **Architecture.** Decoder-only, pre-LM RMSNorm, learned absolute positions, GELU MLP,
  tied embeddings, causally-masked `scaled_dot_product_attention`. Scaled residual init
  on the projection layers, without which the first few hundred steps are spent
  recovering a sane residual scale.
- **Batching.** Length-bucketed, so padding waste stays low on CPU where every padded
  position costs the same as a real one.

## The browser demo

The demo is a single page. Type an English phrase, the model emits cron on WebGPU, and the
page renders the next five fire times computed in the browser. Badges show which execution
provider actually won (WebGPU or WebAssembly), the inlined model size, and per-token and
total latency.

**Weights are inlined into the bundle.** `web/scripts/build.mjs` base64-encodes the
quantized ONNX file into a generated module, so the page has no fetch of its own for the
model and works from `file://`-ish static hosting. The reference implementation this shape
is copied from inlines ~27KB of a hand-written lexer; a transformer does not compress like
that — see "Limitations".

**Failure states are latched and classified**, not retried forever: `no-onnx` when the
weights are missing or unloadable, `no-webgpu` when there is no adapter and the wasm
fallback also failed, `inference-failed` when session creation or a run throws. Each
renders a visible panel rather than a blank output box.

The demo uses the same automaton as training, ported to TypeScript and pinned to the
Python by `grammar/conformance.json` (12,500 state cases and 600 accept/reject cases).

## Limitations

**Quality is what it is.** The numbers are in the table above; the model is a ~25M
parameter network trained from scratch on generated data, and it is not a replacement for
a well-tested cron library. Treat a wrong answer as likely on unusual phrasings.

**The grammar is narrower than cron.** Lists must be written in ascending, non-overlapping
order, so `*/2,5` is rejected by the decoder even though cron-parser accepts it. That is
deliberate (it is what makes duplicate detection checkable character by character, and it
matches how people write lists), but it means the decoder cannot express every valid cron
string, and the guarantee only runs one way: every output is valid cron, but not every
valid cron is reachable.

**Validity is syntactic, satisfiability is checked after the fact.** The automaton rejects
malformed and duplicate-value fields by construction, but it cannot know that day 30 of
February never fires — that needs month/day arithmetic the character-level automaton does
not do. The eval harness verifies every generated expression with cron-parser
(`fires_pct` in the results) rather than repairing it. On random walks through the grammar
about 1% of syntactically valid outputs never fire; the trained model has never produced
one, because that shape is not in its training data.

**No timezones.** Cron has no timezone field. The dataset filters out any phrasing that
mentions one, and the demo renders fire times in the browser's local zone.

**Phrasings are LLM-generated and not individually verified.** The generation prompt is
strict about preserving the schedule, and filters drop timezone/relative-date leakage, but
nothing re-derives the schedule from each phrasing to confirm it still means the same
thing. A wrong paraphrase is a wrong training label. Inspecting samples found them
faithful; that is a spot check, not a proof.

**Two splits share a little data.** Splitting is by expression, but the sampler can reach
the same cron string from two different buckets (a `1-5` day-of-week range is reachable
both as "weekdays" and as a plain range), so 2.8% of val and 2.1% of test expressions also
appear in training. No (phrasing, cron) pair appears in two splits. The effect is small
but it means the held-out numbers are a slight over-estimate.

**The browser bundle is heavy.** `onnxruntime-web` plus its wasm runtime is ~27MB in
`dist/`, far more than the model itself. Inlining the weights was the goal here; a
production deployment would lazy-load the runtime and keep the model as a separate file.

**Two bugs the verification caught, both worth knowing about.** The training labels were
originally aligned with the input rather than shifted by one, which turned the task into
copying the current token — training and validation loss both went to ~0.0 while greedy
decoding produced nothing but `* * * * *`. And the decoder's completion-length estimate
assumed a field dangling on a comma could be closed in one character, which let the length
budget strand the decoder in an empty `allowed()`. Both were invisible to loss curves and
only surfaced because the decoder was exercised end to end and the automaton was tested
against adversarial walks instead of uniform random ones.

## License

MIT.

"""Stage 3: turn the paraphrase checkpoint into the training/eval splits.

Splits come from the canonical rows, so they hold at two levels:

- ``test`` / ``val``: the cron expression *and* its canonical English were never trained
  on. This is the headline number.
- ``holdout``: paraphrases of schedules that *are* in training, one per training
  expression, held back before the model ever saw the rest. This isolates "new way of
  saying a known schedule" from "new schedule".

A phrasing that maps to two different cron expressions is dropped outright rather than
assigned an arbitrary winner — with ranges and lists in the distribution, ``0-2`` and
``0,1,2`` are different strings for the same schedule, and a phrasing that legitimately
fits both would otherwise teach the model to guess.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "out"

PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalize(text: str) -> str:
    return " ".join(PUNCT.sub(" ", text.lower()).split())


def load_checkpoint() -> list[dict]:
    """Each checkpoint line is a batch; the canonical rows are nested inside it."""
    rows: list[dict] = []
    for path in sorted(OUT.glob("paraphrase.ckpt*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.extend(json.loads(line).get("rows", []))
    return rows


def main() -> None:
    rows = load_checkpoint()
    if not rows:
        raise SystemExit("no paraphrase checkpoint found; run data/paraphrase.mjs first")

    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": [], "holdout": []}
    seen: dict[str, str] = {}
    conflicts = 0
    duplicates = 0

    canonical_echoes = 0
    for row in rows:
        phrasings = row.get("phrasings") or []
        if not phrasings:
            continue
        split = row.get("split")
        if split == "train":
            # Last phrasing of a training expression is held back, never trained on.
            targets = [("train", p) for p in phrasings[:-1]] + [("holdout", phrasings[-1])]
        elif split in ("val", "test"):
            targets = [(split, p) for p in phrasings]
        else:
            continue
        # The LLM is asked for *diverse* phrasings and systematically avoids the plainest
        # reading — "every 15 minutes" never appeared for `*/15 * * * *`, so the model had
        # never seen the most obvious way to say the most common schedule. cronstrue's
        # canonical string is itself a valid phrasing, so it is added verbatim.
        targets.append((split if split in ("val", "test") else "train", row["english"]))
        canonical_echoes += 1

        for target_split, text in targets:
            key = normalize(text)
            owner = seen.get(key)
            if owner is not None:
                if owner == row["cron"]:
                    duplicates += 1
                else:
                    conflicts += 1
                continue
            seen[key] = row["cron"]
            buckets[target_split].append(
                {"text": text, "cron": row["cron"], "bucket": row.get("bucket", "unknown")}
            )

    for split, items in buckets.items():
        path = OUT / f"{split}.jsonl"
        with path.open("w") as fh:
            for item in items:
                fh.write(f"{json.dumps(item)}\n")

    total = sum(len(v) for v in buckets.values())
    print(f"assembled {total} pairs from {len(rows)} canonical rows")
    print(f"  dropped {duplicates} repeat phrasings, {conflicts} conflicting phrasings")
    print(f"  added {canonical_echoes} canonical-echo pairs (one per expression)")
    for split, items in buckets.items():
        crons = len({i['cron'] for i in items})
        print(f"  {split:8s} {len(items):7d} pairs  {crons:6d} distinct expressions")
    lens = [len(i["text"]) for i in buckets["train"]]
    lens.sort()
    if lens:
        for pct in (50, 90, 99, 100):
            print(f"  train text length p{pct}: {lens[min(len(lens) - 1, int(len(lens) * pct / 100))]}")
    print("  top buckets (train):")
    for name, count in Counter(i["bucket"] for i in buckets["train"]).most_common(8):
        print(f"    {name:22s} {count}")


if __name__ == "__main__":
    main()

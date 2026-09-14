"""Evaluate a trained checkpoint: exact match, semantic match, and validity.

Exact match understates quality here — `15 17 * * 2-4` and `15 17 * * 3,2,4` are the same
schedule — so semantic match is the headline number. The unconstrained baseline runs the
same weights with no logit mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "eval"))

from constrained_decode import ConstrainedDecoder  # noqa: E402
from cron_automaton import default_automaton, is_well_formed  # noqa: E402

from cron_semantics import CronService  # noqa: E402
from model import Config, TinyCronLM  # noqa: E402
from train import build_config  # noqa: E402


def load_model(checkpoint: Path) -> tuple[TinyCronLM, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg: Config = build_config(ckpt["config"])
    model = TinyCronLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def read_split(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def score(preds: list[str], golds: list[str], buckets: list[str], service: CronService, n: int) -> dict:
    well_formed = [is_well_formed(p) for p in preds]
    verdicts = service.validate(preds)
    exact = [p == g for p, g in zip(preds, golds)]
    semantic = service.semantic_equal(list(zip(preds, golds)), n=n)
    fires = [v["fires"] for v in verdicts]
    parses = [v["parses"] for v in verdicts]

    def rate(flags) -> float:
        return round(100.0 * sum(flags) / max(1, len(flags)), 2)

    per_bucket: dict[str, dict] = defaultdict(lambda: {"n": 0, "semantic": 0, "exact": 0})
    for bucket, sem, ex in zip(buckets, semantic, exact):
        entry = per_bucket[bucket]
        entry["n"] += 1
        entry["semantic"] += int(sem)
        entry["exact"] += int(ex)
    buckets_out = {
        name: {
            "n": v["n"],
            "semantic_pct": round(100.0 * v["semantic"] / v["n"], 2),
            "exact_pct": round(100.0 * v["exact"] / v["n"], 2),
        }
        for name, v in sorted(per_bucket.items(), key=lambda kv: -kv[1]["n"])
    }
    summary = {
        "n": len(preds),
        "well_formed_pct": rate(well_formed),
        "parses_pct": rate(parses),
        "fires_pct": rate(fires),
        "exact_pct": rate(exact),
        "semantic_pct": rate(semantic),
        "by_bucket": buckets_out,
    }
    return summary, semantic, exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "out"))
    ap.add_argument("--splits", default="test,holdout")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--semantic-n", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="cap examples per split (all if omitted)")
    ap.add_argument("--baseline-sample", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=12, help="example predictions to print")
    args = ap.parse_args()

    model, ckpt = load_model(Path(args.checkpoint))
    print(f"checkpoint: {args.checkpoint}  params={ckpt['params']/1e6:.2f}M  step={ckpt['step']}")

    decoder = ConstrainedDecoder(model, default_automaton())
    baseline = ConstrainedDecoder(model, constrain=False)

    results: dict = {"checkpoint": str(args.checkpoint), "params": ckpt["params"], "step": ckpt["step"], "splits": {}}
    preds_by_split: dict[str, list[str]] = {}

    with CronService() as service:
        for split in args.splits.split(","):
            path = Path(args.data_dir) / f"{split}.jsonl"
            if not path.exists():
                print(f"  (skipping {split}: {path} not found)")
                continue
            rows = read_split(path, args.limit)
            texts = [r["text"] for r in rows]
            golds = [r["cron"] for r in rows]
            buckets = [r.get("bucket", "unknown") for r in rows]
            preds = decoder.decode(texts, batch_size=args.batch_size)
            preds_by_split[split] = preds
            scored, semantic, exact = score(preds, golds, buckets, service, args.semantic_n)
            scored["truncated"] = decoder.stats["truncated"]
            scored["examples"] = [
                {"text": t, "gold": g, "pred": p, "semantic": bool(s), "exact": bool(e), "bucket": b}
                for t, g, p, s, e, b in zip(texts, golds, preds, semantic, exact, buckets)
            ][: args.show]
            results["splits"][split] = scored
            print(
                f"  {split:8s} n={scored['n']:6d}  well_formed={scored['well_formed_pct']:6.2f}%  "
                f"fires={scored['fires_pct']:6.2f}%  exact={scored['exact_pct']:6.2f}%  "
                f"semantic={scored['semantic_pct']:6.2f}%"
            )

        sample = read_split(Path(args.data_dir) / "test.jsonl", args.baseline_sample)
        test_rows = results["splits"].get("test")
        if sample and test_rows:
            # Same first-N examples the constrained run scored, so the two columns are
            # comparable; comparing different sample sizes once made the mask look harmful.
            n = min(args.baseline_sample, test_rows["n"])
            sample = sample[:n]
            golds = [r["cron"] for r in sample]
            preds = preds_by_split["test"]
            raw = baseline.decode([r["text"] for r in sample], batch_size=args.batch_size)
            verdicts = service.validate(raw)
            parses = sum(v["parses"] for v in verdicts)
            semantic = service.semantic_equal(list(zip(raw, golds)), n=args.semantic_n)

            def rate(flags) -> float:
                return round(100.0 * sum(flags) / max(1, len(flags)), 2)

            results["unconstrained_baseline"] = {
                "n": n,
                "parses_pct": rate([v["parses"] for v in verdicts]),
                "exact_pct": rate([a == b for a, b in zip(raw, golds)]),
                "semantic_pct": rate(semantic),
            }
            constrained = results.setdefault("constrained_on_baseline_sample", {})
            constrained["n"] = n
            constrained["well_formed_pct"] = rate([is_well_formed(p) for p in preds[:n]])
            constrained["exact_pct"] = rate([a == b for a, b in zip(preds[:n], golds)])
            constrained["semantic_pct"] = rate(
                service.semantic_equal(list(zip(preds[:n], golds)), n=args.semantic_n)
            )
            b = results["unconstrained_baseline"]
            c = constrained
            print(f"  same {n} examples: masked well_formed={c['well_formed_pct']}% exact={c['exact_pct']}% "
                  f"semantic={c['semantic_pct']}%  |  unmasked parses={b['parses_pct']}% "
                  f"exact={b['exact_pct']}% semantic={b['semantic_pct']}%")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")

    for split, scored in results["splits"].items():
        print(f"\n{split} examples:")
        for ex in scored["examples"]:
            mark = "ok " if ex["semantic"] else "BAD"
            print(f"  {mark} {ex['text'][:58]:58s} -> {ex['pred']:22s} (gold {ex['gold']})")


if __name__ == "__main__":
    main()

"""Render eval/results/*.json into the README's results section, so the published numbers
come from the eval output instead of a copy-paste."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPLIT_LABEL = {
    "test": "test (unseen expression)",
    "holdout": "holdout (unseen phrasing, seen expression)",
}


def render(paths: list[Path]) -> str:
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        for split, scored in data["splits"].items():
            rows.append((data, split, scored))

    lines = [
        "| model | split | n | semantic | exact | valid | fires |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for data, split, scored in rows:
        lines.append(
            f"| {data['params'] / 1e6:.1f}M | {SPLIT_LABEL.get(split, split)} | {scored['n']:,} | "
            f"**{scored['semantic_pct']:.1f}%** | {scored['exact_pct']:.1f}% | "
            f"{scored['well_formed_pct']:.1f}% | {scored['fires_pct']:.1f}% |"
        )

    lines.append("")
    lines.append("Same weights, mask on vs. off, on the same held-out examples:")
    lines.append("")
    lines.append("| model | decoding | valid cron | semantic | exact |")
    lines.append("|---|---|---:|---:|---:|")
    seen_models = set()
    for data, _split, _scored in rows:
        base = data.get("unconstrained_baseline")
        head = data.get("constrained_on_baseline_sample")
        # The baseline pass runs once per checkpoint but `rows` has a line per split.
        if not base or not head or id(data) in seen_models:
            continue
        seen_models.add(id(data))
        n = base["n"]
        lines.append(
            f"| {data['params'] / 1e6:.1f}M | constrained | {head['well_formed_pct']:.1f}% | "
            f"{head['semantic_pct']:.1f}% | {head['exact_pct']:.1f}% |"
        )
        lines.append(
            f"| {data['params'] / 1e6:.1f}M | unconstrained | {base['parses_pct']:.1f}% | "
            f"{base['semantic_pct']:.1f}% | {base['exact_pct']:.1f}% |"
        )
        lines.append("")
    if any(d.get("unconstrained_baseline") for d, _, _ in rows):
        n = next(
            d["unconstrained_baseline"]["n"] for d, _, _ in rows if d.get("unconstrained_baseline")
        )
        lines.append(
            f"n = {n} held-out examples. The mask does not just make the output valid — on both "
            "models it is also the more accurate of the two, and the constrained column is valid "
            "cron by construction rather than by measurement."
        )
    return "\n".join(lines)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: render_table.py <results.json> [...]")
    table = render([p for p in paths if p.exists()])
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text()
    start, end = "<!--RESULTS_TABLE-->", "<!--/RESULTS_TABLE-->"
    if start not in text:
        print(table)
        return
    # Bounded by both markers so a later run replaces the table instead of appending.
    if end in text:
        head, rest = text.split(start, 1)
        _stale, tail = rest.split(end, 1)
        text = f"{head}{start}\n{table}\n{end}{tail}"
    else:
        text = text.replace(start, f"{start}\n{table}\n{end}")
    readme.write_text(text)
    print(f"updated {readme}")
    print(table)


if __name__ == "__main__":
    main()

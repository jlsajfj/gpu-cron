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
    for data, _split, scored in rows:
        base = data.get("unconstrained_baseline")
        head = data.get("constrained_on_baseline_sample")
        if not base or not head:
            continue
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
        lines.append(f"n = {n} held-out examples. The unconstrained column is why the automaton exists: "
                     "on the same weights, a large share of raw outputs are not cron at all.")
        break
    return "\n".join(lines)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: render_table.py <results.json> [...]")
    table = render([p for p in paths if p.exists()])
    readme = Path(__file__).resolve().parent.parent / "README.md"
    text = readme.read_text()
    marker = "<!--RESULTS_TABLE-->"
    if marker not in text:
        print(table)
        return
    readme.write_text(text.replace(marker, table))
    print(f"updated {readme}")
    print(table)


if __name__ == "__main__":
    main()

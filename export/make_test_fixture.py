"""Generate the committed fixture that pins web/src/forward.ts to the numpy reference.

A tiny randomly-initialised model is enough: the point is that the two forward passes agree
on the same weights, which says nothing about accuracy and everything about whether the
hand-written TypeScript still matches. export_js.py separately checks the numpy reference
against torch, so JS == numpy == torch is a chain rather than three separate hopes.

    bin/py export/make_test_fixture.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from model import TinyCronLM  # noqa: E402
from train import build_config  # noqa: E402

FIXTURE = ROOT / "web" / "test" / "fixtures"
CONFIG = {"d_model": 32, "n_layer": 2, "n_head": 2, "d_ff": 64, "max_len": 192, "dropout": 0.0}


def main() -> None:
    torch.manual_seed(0)
    model = TinyCronLM(build_config(CONFIG))
    FIXTURE.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "checkpoint.pt"
        torch.save(
            {"model": model.state_dict(), "config": CONFIG, "step": 0, "params": 0},
            checkpoint,
        )
        subprocess.run(
            [
                str(ROOT / "bin" / "py"),
                str(ROOT / "export" / "export_js.py"),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(FIXTURE / "tiny"),
                "--dtype",
                "fp32",
                "--reference-out",
                str(FIXTURE / "tiny-reference.json"),
            ],
            check=True,
            cwd=ROOT,
        )

    manifest = json.loads((FIXTURE / "tiny.json").read_text())
    reference = json.loads((FIXTURE / "tiny-reference.json").read_text())
    total = sum(
        entry["shape"][0] * (entry["shape"][1] if len(entry["shape"]) > 1 else 1)
        for entry in manifest["tensors"]
    )
    size_kb = (FIXTURE / "tiny.bin").stat().st_size / 1024
    print(f"fixture: {total} params, {len(reference['cases'])} reference cases, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()

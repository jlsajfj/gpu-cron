"""The only test over the whole path: data, training loop, checkpoint, constrained decode."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "train"))

PY = ROOT / "bin" / "py"

SYNTHETIC = [
    ("every day at 9am", "0 9 * * *"),
    ("every 5 minutes", "*/5 * * * *"),
    ("weekdays at 6:30 in the evening", "30 18 * * 1-5"),
    ("on the first of the month at midnight", "0 0 1 * *"),
    ("every hour on Sundays", "0 * * 0"),
]


class TestTrainingSmoke(unittest.TestCase):
    def test_train_then_decode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            data_dir.mkdir()
            rows = [{"text": t, "cron": c, "bucket": "smoke"} for t, c in SYNTHETIC * 4]
            for split in ("train", "val"):
                (data_dir / f"{split}.jsonl").write_text(
                    "\n".join(json.dumps(r) for r in rows) + "\n"
                )
            out_dir = Path(tmp) / "run"

            proc = subprocess.run(
                [
                    str(PY),
                    str(ROOT / "train" / "train.py"),
                    "--config",
                    str(ROOT / "configs" / "smoke.json"),
                    "--data-dir",
                    str(data_dir),
                    "--out",
                    str(out_dir),
                    "--steps",
                    "20",
                    "--warmup",
                    "5",
                    "--batch-size",
                    "8",
                    "--threads",
                    "4",
                    "--log-every",
                    "10",
                    "--save-every",
                    "10",
                ],
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
            self.assertEqual(proc.returncode, 0, f"training failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("saved", proc.stdout)
            checkpoint = out_dir / "checkpoint.pt"
            self.assertTrue(checkpoint.exists())

            import torch

            from constrained_decode import ConstrainedDecoder
            from cron_automaton import is_well_formed
            from model import TinyCronLM
            from train import build_config

            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model = TinyCronLM(build_config(ckpt["config"]))
            model.load_state_dict(ckpt["model"])

            prompts = [t for t, _ in SYNTHETIC]
            decoder = ConstrainedDecoder(model)
            preds = decoder.decode(prompts, batch_size=8)

            self.assertEqual(len(preds), len(prompts))
            for text, pred in zip(prompts, preds):
                self.assertTrue(is_well_formed(pred), f"{text!r} -> {pred!r} is not well-formed cron")
                self.assertEqual(len(pred.split(" ")), 5, f"{text!r} -> {pred!r}")
            self.assertEqual(decoder.stats["truncated"], 0, "encoding is far too long")

    def test_labels_are_next_token_shifted(self) -> None:
        """labels[i] must be ids[i+1]: lined up with ids[i] the task is copying, not mapping."""
        from data import encode_example

        ids, labels = encode_example("every day at 9am", "0 9 * * *")
        self.assertEqual(len(ids), len(labels))
        labelled = [i for i, l in enumerate(labels) if l != -100]
        self.assertTrue(labelled, "no supervised positions")
        for i in labelled:
            self.assertEqual(labels[i], ids[i + 1], f"label at {i} is not ids[i+1]")
        self.assertEqual(chr(labels[labelled[0]]), "0")
        self.assertEqual(labels[-1], -100, "nothing to predict after the end")

    def test_batched_decode_matches_one_at_a_time(self) -> None:
        """Padding must not change an answer.

        A batched decode of unequal-length prompts pads them, and the model never saw a real
        token follow a PAD in training, so a padded batch can quietly decode differently from
        the one-row-at-a-time case the browser runs. Prompt lengths here are deliberately
        spread out; the invariant holds for any weights, so no checkpoint is needed.
        """
        import torch

        from constrained_decode import ConstrainedDecoder
        from cron_automaton import is_well_formed
        from model import Config, TinyCronLM

        torch.manual_seed(0)
        model = TinyCronLM(Config(d_model=64, n_layer=2, n_head=2, d_ff=128, max_len=192))
        texts = [
            "every 5 minutes",
            "at 9:15 in the morning on the first day of every month",
            "weekdays",
            "run this at half past six in the evening on Tuesdays and Thursdays",
            "midnight",
            "every third hour from 1 AM to 1 PM, Monday to Friday",
        ]
        one = ConstrainedDecoder(model).decode(texts, batch_size=1)
        many = ConstrainedDecoder(model).decode(texts, batch_size=len(texts))
        self.assertEqual(one, many)
        for text, pred in zip(texts, many):
            self.assertTrue(is_well_formed(pred), f"{text!r} -> {pred!r}")

    def test_lr_schedule_shape(self) -> None:
        from train import lr_at

        peak, warmup, total = 1e-3, 100, 1000
        self.assertLess(lr_at(0, peak=peak, warmup=warmup, total=total), peak)
        self.assertAlmostEqual(lr_at(warmup - 1, peak=peak, warmup=warmup, total=total), peak)
        self.assertLess(lr_at(total, peak=peak, warmup=warmup, total=total), peak * 0.1)
        mid = lr_at(warmup + (total - warmup) // 2, peak=peak, warmup=warmup, total=total)
        self.assertGreater(mid, lr_at(total, peak=peak, warmup=warmup, total=total))


if __name__ == "__main__":
    unittest.main()

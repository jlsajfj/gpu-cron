"""Byte-level encoding and length-bucketed batching.

The prompt is ``"<english> => "`` and the target is the cron string followed by EOS, so
the model only ever has to learn the mapping, not a formatting convention that varies.
Loss is taken on the cron bytes and EOS alone; the prompt is masked out.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from model import BOS, EOS, PAD

PROMPT_SUFFIX = " => "


def encode_example(text: str, cron: str) -> tuple[list[int], list[int]]:
    """Return (ids, labels) for next-token training.

    `labels[i]` is the token to predict *at* position i, which is `ids[i + 1]` — the loss
    pairs `logits[i]` with `labels[i]`, and a causal model at position i can see `ids[i]`
    but not `ids[i + 1]`. Lining labels up with `ids[i]` instead turns this into a copy
    task the model can solve perfectly without learning anything, which is exactly what an
    earlier version did: near-zero training and validation loss, and total failure under
    greedy decoding.
    """
    prompt = (text + PROMPT_SUFFIX).encode("utf-8")
    target = list(cron.encode("utf-8")) + [EOS]
    ids = [BOS, *prompt, *target]
    labels = [-100] * len(ids)
    for k, token in enumerate(target):
        labels[len(prompt) + k] = token
    return ids, labels


@dataclass
class Example:
    ids: list[int]
    labels: list[int]

    @property
    def length(self) -> int:
        return len(self.ids)


def load_split(path: Path, max_len: int) -> tuple[list[Example], int]:
    examples: list[Example] = []
    skipped = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ids, labels = encode_example(row["text"], row["cron"])
        if len(ids) > max_len:
            skipped += 1
            continue
        examples.append(Example(ids, labels))
    return examples, skipped


def collate(batch: list[Example], pad_to: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    width = pad_to or max(e.length for e in batch)
    idx = torch.full((len(batch), width), PAD, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for row, example in enumerate(batch):
        n = example.length
        idx[row, :n] = torch.tensor(example.ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(example.labels, dtype=torch.long)
    return idx, labels


class LengthBucketedBatcher:
    """Batches similar-length examples together so padding waste stays low on CPU, where
    every padded position costs the same as a real one.

    Sorting by length alone is not enough: the dataset is written expression-by-expression,
    so the 8-40 phrasings of one cron are adjacent and near-identical in length. Contiguous
    chunks then hand the model a batch with a single target, and the loss collapses to
    memorisation within a few dozen steps. Shuffling inside a window large enough to mix
    expressions keeps padding bounded while restoring a real gradient signal.
    """

    def __init__(self, examples: list[Example], batch_size: int, seed: int = 0, pad_multiple: int = 8):
        self.examples = examples
        self.batch_size = batch_size
        self.pad_multiple = pad_multiple
        self.rng = random.Random(seed)

    def _windows(self):
        order = sorted(range(len(self.examples)), key=lambda i: self.examples[i].length)
        span = self.batch_size * 50
        for start in range(0, len(order), span):
            window = order[start : start + span]
            self.rng.shuffle(window)
            yield window

    def epochs(self):
        while True:
            batches = []
            for window in self._windows():
                for i in range(0, len(window), self.batch_size):
                    batches.append(window[i : i + self.batch_size])
            self.rng.shuffle(batches)
            for batch in batches:
                picked = [self.examples[i] for i in batch]
                width = max(e.length for e in picked)
                width = -(-width // self.pad_multiple) * self.pad_multiple
                yield collate(picked, width)

    def steps_per_epoch(self) -> int:
        """Must match the number of batches `epochs()` actually yields, or a resumed run
        skips the wrong distance into the stream."""
        return -(-len(self.examples) // self.batch_size)

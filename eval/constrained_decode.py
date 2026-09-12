"""Constrained decoding: logits are masked by the cron automaton at every step.

Nothing here repairs a string after the fact. At each position the automaton reports the
characters that keep the output on a path to a complete cron expression, every other
logit is set to -inf, and the argmax is taken over what remains. A decoded string is
therefore well-formed cron by construction — `is_well_formed` holds for 100% of outputs
by the same argument that makes it hold for the empty prefix, not by luck.

The same loop runs in the browser (web/src/decode.js) against the ONNX export, which is
why the automaton exists twice and why grammar/conformance.json pins the two together.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import BOS, EOS, PAD  # noqa: E402

from cron_automaton import CronAutomaton, default_automaton  # noqa: E402

PROMPT_SUFFIX = " => "


class ConstrainedDecoder:
    def __init__(
        self,
        model,
        automaton: CronAutomaton | None = None,
        max_new: int = 72,
        temperature: float = 0.0,
        top_k: int = 0,
        constrain: bool = True,
    ):
        self.model = model
        self.constrain = constrain
        self.automaton = automaton or default_automaton()
        self.max_new = max_new
        self.temperature = temperature
        self.top_k = top_k
        self._mask_cache: dict[tuple[int, str, int], torch.Tensor] = {}
        self.vocab_size = model.head.out_features
        self.stats = {"truncated": 0}

    def _allowed_ids(self, state) -> list[int]:
        allowed = self.automaton.allowed(state)
        ids = []
        for ch in allowed:
            ids.append(EOS if ch == "\x00" else ord(ch))
        return sorted(set(ids))

    def _mask(self, state, device) -> torch.Tensor:
        """Boolean (vocab,) tensor: True where the token is legal in `state`."""
        if not self.constrain:
            return torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        cached = self._mask_cache.get(state)
        if cached is not None:
            return cached
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        mask[self._allowed_ids(state)] = True
        self._mask_cache[state] = mask
        return mask

    @torch.no_grad()
    def decode(self, texts: list[str], batch_size: int = 64, progress=None) -> list[str]:
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            out.extend(self._decode_batch(texts[i : i + batch_size]))
            if progress is not None:
                progress(min(len(texts), i + batch_size), len(texts))
        return out

    @torch.no_grad()
    def _decode_batch(self, texts: list[str]) -> list[str]:
        device = next(self.model.parameters()).device
        was_training = self.model.training
        self.model.eval()

        prompts = [[BOS, *(t + PROMPT_SUFFIX).encode("utf-8")] for t in texts]
        lengths = [len(p) for p in prompts]
        width = max(lengths)
        idx = torch.full((len(prompts), width), PAD, dtype=torch.long, device=device)
        for row, prompt in enumerate(prompts):
            idx[row, : len(prompt)] = torch.tensor(prompt, dtype=torch.long, device=device)

        states = [self.automaton.start() for _ in texts]
        emitted: list[list[str]] = [[] for _ in texts]
        done = [False] * len(texts)

        logits = self.model(idx)  # (batch, width, vocab)
        last = logits[torch.arange(len(prompts)), torch.tensor(lengths, device=device) - 1, :]
        step_logits = last

        for _ in range(self.max_new):
            if all(done):
                break
            masked = step_logits.clone()
            for row, state in enumerate(states):
                if done[row]:
                    # Park finished rows on EOS so the batch stays rectangular.
                    row_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
                    row_mask[EOS] = True
                else:
                    row_mask = self._mask(state, device)
                masked[row] = masked[row].masked_fill(~row_mask, float("-inf"))
            nxt = self._pick(masked)
            for row, token in enumerate(nxt.tolist()):
                if done[row]:
                    continue
                if token == EOS:
                    done[row] = True
                    continue
                if not self.constrain:
                    # Unconstrained runs measure what the raw model emits, so the automaton
                    # is not consulted at all — walking it here would reject the illegal
                    # characters this mode exists to observe.
                    emitted[row].append(chr(token))
                    continue
                ch = chr(token)
                states[row] = self.automaton.advance(states[row], ch)
                if states[row] is None:
                    raise AssertionError(f"automaton rejected its own allowed character {ch!r}")
                emitted[row].append(ch)
            idx = torch.cat([idx, nxt.view(-1, 1)], dim=1)
            step_logits = self.model(idx[:, -self.model.cfg.max_len :])[:, -1, :]

        # Running out of budget is a real outcome (the model can emit a long list), so it
        # is counted and reported rather than asserted away.
        self.stats["truncated"] += done.count(False)
        self.model.train(was_training)
        return ["".join(chars) for chars in emitted]

    def _pick(self, masked: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0:
            return masked.argmax(dim=-1)
        scaled = masked / self.temperature
        if self.top_k > 0:
            k = min(self.top_k, scaled.size(-1))
            cutoff = torch.topk(scaled, k, dim=-1).values[:, -1:]
            scaled = scaled.masked_fill(scaled < cutoff, float("-inf"))
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

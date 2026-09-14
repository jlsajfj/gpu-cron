"""GPU path: LoRA fine-tune of SmolLM2-135M instead of training from scratch.

Unverified: never executed on the box (no GPU), and nothing in the tested path imports it.
SmolLM2's tokenizer is BPE, so only tokens decoding to exactly one cron character can be
allowed — decoding stays valid by construction, but one character per token is slow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

from cron_automaton import EOS, default_automaton  # noqa: E402

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
PROMPT_SUFFIX = " => "


def require_gpu_stack():
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as err:  # pragma: no cover - exercised only on a GPU box
        raise SystemExit(
            f"missing dependency: {err}. Install with `pip install -r requirements-gpu.txt`, "
            "and note this path needs a GPU (the box does not have one)."
        ) from err
    return torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model


def load_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def build_examples(rows, tokenizer, max_len: int):
    out = []
    for row in rows:
        prompt = tokenizer(f"{row['text'].lower()}{PROMPT_SUFFIX}", add_special_tokens=False)["input_ids"]
        target = tokenizer(row["cron"], add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        ids = prompt + target
        if len(ids) > max_len:
            continue
        labels = [-100] * len(prompt) + target
        out.append((ids, labels))
    return out


def allowed_token_mask(automaton, state, tokenizer, device):
    import torch

    allowed = automaton.allowed(state)
    ids = []
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id])
        if text == EOS or (len(text) == 1 and text in allowed):
            ids.append(token_id)
    mask = torch.full((len(tokenizer),), float("-inf"), device=device)
    if ids:
        mask[torch.tensor(ids, device=device)] = 0.0
    return mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "out"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--sample", type=int, default=8, help="constrained samples to print after training")
    args = ap.parse_args()

    torch, AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model = require_gpu_stack()
    if not torch.cuda.is_available():
        print("warning: no CUDA device visible; this path is intended for a GPU box", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r * 2,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    examples = build_examples(load_rows(Path(args.data_dir) / "train.jsonl", args.limit), tokenizer, args.max_len)
    print(f"{len(examples)} training examples", flush=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = torch.Generator().manual_seed(0)

    def batches():
        order = torch.randperm(len(examples), generator=rng).tolist()
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            chunk = [examples[j] for j in order[i : i + args.batch_size]]
            width = max(len(ids) for ids, _ in chunk)
            idx = torch.full((len(chunk), width), tokenizer.pad_token_id or 0, device=device)
            labels = torch.full((len(chunk), width), -100, device=device)
            for row, (ids, labs) in enumerate(chunk):
                idx[row, : len(ids)] = torch.tensor(ids, device=device)
                labels[row, : len(labs)] = torch.tensor(labs, device=device)
            yield idx, labels

    stream = batches()
    for step in range(1, args.steps + 1):
        try:
            idx, labels = next(stream)
        except StopIteration:
            stream = batches()
            idx, labels = next(stream)
        loss = model(input_ids=idx, labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == 1:
            print(json.dumps({"step": step, "loss": round(loss.item(), 4)}), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"saved adapter to {out_dir}", flush=True)

    if args.sample:
        automaton = default_automaton()
        model.eval()
        rows = load_rows(Path(args.data_dir) / "test.jsonl", args.sample)
        with torch.no_grad():
            for row in rows:
                ids = tokenizer(f"{row['text'].lower()}{PROMPT_SUFFIX}", return_tensors="pt")["input_ids"].to(device)
                state = automaton.start()
                for _ in range(64):
                    logits = model(input_ids=ids).logits[0, -1, :]
                    logits = logits + allowed_token_mask(automaton, state, tokenizer, logits.device)
                    token = int(logits.argmax())
                    text = tokenizer.decode([token])
                    if token == tokenizer.eos_token_id or state is None:
                        break
                    if len(text) != 1:
                        break
                    state = automaton.advance(state, text)
                    ids = torch.cat([ids, torch.tensor([[token]], device=device)], dim=1)
                print(f"  {row['text'][:50]:50s} -> gold {row['cron']}", flush=True)


if __name__ == "__main__":
    main()

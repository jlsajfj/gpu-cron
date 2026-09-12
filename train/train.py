from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data import LengthBucketedBatcher, load_split
from model import VOCAB_SIZE, Config, TinyCronLM


def build_config(raw: dict) -> Config:
    return Config(
        d_model=raw.get("d_model", 512),
        n_layer=raw.get("n_layer", 8),
        n_head=raw.get("n_head", 8),
        d_ff=raw.get("d_ff", 2048),
        max_len=raw.get("max_len", 192),
        dropout=raw.get("dropout", 0.0),
    )


def lr_at(step: int, *, peak: float, warmup: int, total: int, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (min_ratio + (1 - min_ratio) * cosine)


def evaluate_val(model, examples, batch_size: int = 64, max_examples: int = 2000) -> float:
    """Mean loss on a fixed val slice, in eval mode.

    Training loss is measured on the batch just consumed and says nothing about
    generalisation on a task this structured; this is the number worth trusting.
    """
    import torch.nn.functional as F

    from data import collate

    subset = examples[:max_examples]
    was_training = model.training
    model.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(subset), batch_size):
            idx, labels = collate(subset[i : i + batch_size])
            logits = model(idx)
            total += F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100
            ).item()
    model.train(was_training)
    return total / max(1, -(-len(subset) // batch_size))


def save(path: Path, model, optimizer, raw: dict, step: int, params: int, log: list) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": raw,
            "step": step,
            "params": params,
            "log": log,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default="data/out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--time-budget-min", type=float, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run-steps", type=int, default=0)
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="bfloat16 autocast for the training step; ~1.6x faster on ARM cores with bf16",
    )
    args = ap.parse_args()

    raw = json.loads(Path(args.config).read_text())
    cfg = build_config(raw)
    steps = args.steps or raw.get("steps", 8000)
    batch_size = args.batch_size or raw.get("batch_size", 64)
    lr = args.lr or raw.get("lr", 3e-3)

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    data_dir = Path(args.data_dir)
    train_examples, train_skipped = load_split(data_dir / "train.jsonl", cfg.max_len)
    val_examples, val_skipped = load_split(data_dir / "val.jsonl", cfg.max_len)
    if not train_examples:
        raise SystemExit(f"no training examples in {data_dir/'train.jsonl'}")

    model = TinyCronLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    lengths = sorted(len(e.ids) for e in train_examples)
    mean_len = sum(lengths) / len(lengths)
    header = {
        "config": raw,
        "params": n_params,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "skipped_too_long": {"train": train_skipped, "val": val_skipped},
        "mean_seq_len": round(mean_len, 1),
        "p99_seq_len": lengths[int(len(lengths) * 0.99)],
        "batch_size": batch_size,
        "steps": steps,
        "lr": lr,
        "bf16": args.bf16,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(header, indent=2), flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    batcher = LengthBucketedBatcher(train_examples, batch_size, seed=args.seed)
    steps_per_epoch = max(1, batcher.steps_per_epoch())
    epochs = steps / steps_per_epoch
    print(f"steps/epoch={steps_per_epoch}  epochs={epochs:.2f}", flush=True)

    stream = batcher.epochs()
    start_step = 0
    log: list[dict] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        log = ckpt.get("log", [])
        for _ in range(start_step * steps_per_epoch):
            next(stream)
        print(f"resumed from {args.resume} at step {start_step}", flush=True)

    if args.dry_run_steps:
        model.train()
        t0 = time.time()
        tokens = 0
        for _ in range(args.dry_run_steps):
            idx, labels = next(stream)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=args.bf16):
                _, loss = model(idx, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            tokens += idx.numel()
        dt = time.time() - t0
        per_step = dt / args.dry_run_steps
        print(
            f"dry run: {per_step:.2f}s/step  {tokens / dt:.0f} tok/s  "
            f"-> {steps} steps would take {steps * per_step / 3600:.1f}h",
            flush=True,
        )
        return

    started = time.time()
    deadline = started + args.time_budget_min * 60 if args.time_budget_min else None
    tokens_seen = 0
    model.train()
    step = start_step
    running_loss = 0.0
    running_count = 0

    while step < steps:
        idx, labels = next(stream)
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, peak=lr, warmup=args.warmup, total=steps)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=args.bf16):
            _, loss = model(idx, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens_seen += idx.numel()
        running_loss += loss.item()
        running_count += 1
        step += 1

        if step % args.log_every == 0 or step == steps:
            elapsed = time.time() - started
            avg = running_loss / max(1, running_count)
            record = {
                "step": step,
                "loss": round(avg, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 6),
                "elapsed_s": round(elapsed, 1),
                "tok_per_s": round(tokens_seen / elapsed),
                "epoch": round(step / steps_per_epoch, 3),
            }
            log.append(record)
            print(json.dumps(record), flush=True)
            running_loss = 0.0
            running_count = 0

        if step % args.save_every == 0:
            val_loss = evaluate_val(model, val_examples, batch_size)
            log.append({"step": step, "val_loss": round(val_loss, 4)})
            print(json.dumps({"step": step, "val_loss": round(val_loss, 4)}), flush=True)
            save(out_dir / "checkpoint.pt", model, optimizer, raw, step - 1, n_params, log)

        if deadline and time.time() > deadline and step < steps:
            print(f"time budget hit at step {step}/{steps}", flush=True)
            break

    # Full-length val pass at the end, so the step budget isn't the only evidence.
    for limit in (2000, 20000):
        if len(val_examples) > limit:
            continue
        full = evaluate_val(model, val_examples, batch_size, max_examples=limit)
        print(json.dumps({"step": step, "val_loss_full": round(full, 4)}), flush=True)
        log.append({"step": step, "val_loss_full": round(full, 4)})
        break
    save(out_dir / "checkpoint.pt", model, optimizer, raw, step - 1, n_params, log)
    (out_dir / "train_log.json").write_text(json.dumps({"header": header, "log": log}, indent=2))
    print(f"saved {out_dir/'checkpoint.pt'} after {step} steps", flush=True)


if __name__ == "__main__":
    main()

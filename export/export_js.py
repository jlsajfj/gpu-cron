"""Export a checkpoint as int8 weights a plain JavaScript forward pass can read.

ONNX Runtime Web is ~26 MB of wasm, which is more than this model and its whole dataset
put together — for a few hundred thousand parameters a runtime is pure download cost. The
matrices here are small enough that typed-array loops in the page are fast (tens of
milliseconds per decode), so the model ships as raw bytes plus a manifest and web/src/
implements the forward pass directly.

Weights are quantized per output row to int8 with a float32 scale per row, which keeps the
error far below what a per-tensor scale would. RMSNorm gains stay in float32 — they are a
few hundred numbers and they scale the residual stream.

    bin/py export/export_js.py --checkpoint runs/nano/checkpoint.pt --out web/weights/model
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from model import TinyCronLM  # noqa: E402
from train import build_config  # noqa: E402

# (name in the manifest, attribute path in the state dict) for the float32 tensors.
FP32_SUFFIXES = ("norm1.weight", "norm2.weight", "norm_f.weight")


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * weight


def gelu(x: np.ndarray) -> np.ndarray:
    c = np.float32(np.sqrt(2.0 / np.pi))
    return (0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))).astype(np.float32)


def forward(weights: dict, idx: list[int], n_head: int, eps: float = 1e-6) -> np.ndarray:
    """Mirror of web/src/forward.ts. Kept structurally identical so a mismatch is a bug in
    one of the two, not a difference in how the arithmetic was grouped."""
    d_model = weights["tok.weight"].shape[1]
    head_dim = d_model // n_head
    t = len(idx)

    x = weights["tok.weight"][idx] + weights["pos.weight"][:t]
    mask = np.triu(np.full((t, t), -np.inf, dtype=np.float32), 1)

    for layer in range(weights["_n_layer"]):
        p = f"blocks.{layer}."
        h = rms_norm(x, weights[p + "norm1.weight"], eps)
        qkv = h @ weights[p + "attn.qkv.weight"].T
        q, k, v = np.split(qkv, 3, axis=-1)
        shape = (t, n_head, head_dim)
        q = q.reshape(shape).transpose(1, 0, 2)
        k = k.reshape(shape).transpose(1, 0, 2)
        v = v.reshape(shape).transpose(1, 0, 2)
        scores = (q @ k.transpose(0, 2, 1)) / np.float32(np.sqrt(head_dim)) + mask
        scores = scores - scores.max(-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(-1, keepdims=True)
        attn = (probs @ v).transpose(1, 0, 2).reshape(t, d_model)
        x = x + attn @ weights[p + "attn.proj.weight"].T

        h = rms_norm(x, weights[p + "norm2.weight"], eps)
        x = x + gelu(h @ weights[p + "mlp.fc.weight"].T) @ weights[p + "mlp.proj.weight"].T

    h = rms_norm(x, weights["norm_f.weight"], eps)
    return h @ weights["tok.weight"].T  # tied embeddings


def quantize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.abs(matrix).max(axis=1) / 127.0
    scales[scales == 0] = 1.0
    q = np.rint(matrix / scales[:, None]).clip(-127, 127).astype(np.int8)
    return q, scales.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="path stem; writes <stem>.bin and <stem>.json")
    ap.add_argument("--layers", type=int, default=None, help="export only the first N blocks")
    ap.add_argument(
        "--reference-out",
        default=None,
        help="also write the numpy forward's logits, so a test can pin the JS to them",
    )
    ap.add_argument(
        "--dtype",
        choices=("fp32", "int8"),
        default="fp32",
        help="fp32 is exact and, at a few hundred thousand parameters, only ~1 MB",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cfg = build_config(ckpt["config"])
    model = TinyCronLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    state = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    n_layer = args.layers or cfg.n_layer

    manifest: list[dict] = []
    blob = bytearray()

    dtype = args.dtype

    def add_matrix(name: str, matrix: np.ndarray) -> None:
        if dtype == "fp32":
            add_f32(name, matrix.reshape(-1), shape=list(matrix.shape))
        else:
            add_i8(name, matrix)

    def add_i8(name: str, matrix: np.ndarray) -> None:
        q, scales = quantize(matrix)
        offset = len(blob)
        blob.extend(q.tobytes())
        # The page views the scales as Float32Array over the same buffer, so every tensor
        # has to start on a 4-byte boundary.
        blob.extend(b"\x00" * (-len(blob) % 4))
        scale_offset = len(blob)
        blob.extend(scales.tobytes())
        manifest.append(
            {
                "name": name,
                "kind": "i8",
                "shape": list(matrix.shape),
                "offset": offset,
                "scaleOffset": scale_offset,
            }
        )

    def add_f32(name: str, vector: np.ndarray, shape: list[int] | None = None) -> None:
        blob.extend(b"\x00" * (-len(blob) % 4))
        offset = len(blob)
        blob.extend(np.ascontiguousarray(vector, dtype=np.float32).tobytes())
        manifest.append(
            {"name": name, "kind": "f32", "shape": shape or list(vector.shape), "offset": offset}
        )

    add_matrix("tok.weight", state["tok.weight"])
    add_matrix("pos.weight", state["pos.weight"])
    for layer in range(n_layer):
        p = f"blocks.{layer}."
        add_matrix(p + "attn.qkv.weight", state[p + "attn.qkv.weight"])
        add_matrix(p + "attn.proj.weight", state[p + "attn.proj.weight"])
        add_matrix(p + "mlp.fc.weight", state[p + "mlp.fc.weight"])
        add_matrix(p + "mlp.proj.weight", state[p + "mlp.proj.weight"])
        add_f32(p + "norm1.weight", state[p + "norm1.weight"])
        add_f32(p + "norm2.weight", state[p + "norm2.weight"])
    add_f32("norm_f.weight", state["norm_f.weight"])

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    (out_stem.with_suffix(".bin")).write_bytes(bytes(blob))
    (out_stem.with_suffix(".json")).write_text(
        json.dumps(
            {
                "d_model": cfg.d_model,
                "n_layer": n_layer,
                "n_head": cfg.n_head,
                "d_ff": cfg.d_ff,
                "max_len": cfg.max_len,
                "vocab": 259,
                "dtype": dtype,
                "tensors": manifest,
            },
            indent=1,
        )
    )

    # Verify by dequantizing and re-running the forward, rather than trusting the layout.
    def rebuild() -> dict:
        raw = (out_stem.with_suffix(".bin")).read_bytes()
        w: dict = {"_n_layer": n_layer}
        for entry in manifest:
            if entry["kind"] == "i8":
                rows, cols = entry["shape"]
                q = np.frombuffer(raw, np.int8, rows * cols, entry["offset"]).reshape(rows, cols)
                s = np.frombuffer(raw, np.float32, rows, entry["scaleOffset"])
                w[entry["name"]] = q.astype(np.float32) * s[:, None]
            else:
                n = int(np.prod(entry["shape"]))
                w[entry["name"]] = (
                    np.frombuffer(raw, np.float32, n, entry["offset"]).copy().reshape(entry["shape"])
                )
        return w

    weights = rebuild()
    prompts = [
        "every weekday at 9am",
        "every 15 minutes",
        "the first of the month at midnight",
        "9:45 pm on July 15",
        "every third hour from 1 AM to 1 PM, Monday to Friday",
    ]
    worst = 0.0
    dumped: list[dict] = []
    for text in prompts:
        ids = [257, *(text.lower() + " => ").encode()]
        with torch.no_grad():
            torch_logits = model(torch.tensor([ids]))[0, -1].numpy()
        got = forward(weights, ids, cfg.n_head)[-1]
        diff = float(np.abs(torch_logits - got).max())
        worst = max(worst, diff)
        print(
            f"  {text[:34]:36s} max|Δlogit| {diff:.4f}  "
            f"argmax {'match' if torch_logits.argmax() == got.argmax() else 'DIFFER'}"
        )
        dumped.append({"ids": ids, "logits": [round(float(v), 5) for v in got]})

    if args.reference_out:
        Path(args.reference_out).write_text(
            json.dumps({"dtype": dtype, "n_head": cfg.n_head, "cases": dumped})
        )
        print(f"wrote {args.reference_out}")

    print(f"wrote {out_stem.with_suffix('.bin')} ({(out_stem.with_suffix('.bin')).stat().st_size / 1024:.0f} KB)")
    print(f"wrote {out_stem.with_suffix('.json')} ({len(manifest)} tensors)")
    print(f"worst max|Δlogit| vs torch: {worst:.4f}")


if __name__ == "__main__":
    main()

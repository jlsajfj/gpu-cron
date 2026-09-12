"""Export a trained checkpoint to ONNX for the browser decoder.

The graph returns only the last position's logits and keeps the sequence length dynamic —
the browser re-runs the growing prefix each step rather than keeping a KV cache. A trailing
``.int8`` in ``--out`` names the quantized file, so the float export lands beside it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from model import VOCAB_SIZE, TinyCronLM  # noqa: E402
from train import build_config  # noqa: E402


class LastPositionLogits(nn.Module):
    def __init__(self, model: TinyCronLM):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)[:, -1, :]


def load_model(checkpoint: Path) -> tuple[TinyCronLM, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = build_config(ckpt["config"])
    model = TinyCronLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def export_fp32(wrapper: nn.Module, out: Path, opset: int) -> None:
    dummy = torch.randint(0, 256, (1, 24), dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["ids"],
        output_names=["logits"],
        dynamic_axes={"ids": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # The browser inlines one file, so the weights cannot live in a sibling .data blob.
        external_data=False,
    )


def describe_graph(path: Path) -> str:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    opset = model.opset_import[0].version
    shapes = []
    for value in [*model.graph.input, *model.graph.output]:
        dims = [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]
        shapes.append(f"{value.name}{dims}")
    return f"opset {opset}  " + " ".join(shapes)


def compare(path: Path, wrapper: nn.Module, batches: list[torch.Tensor]) -> float:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for i, ids in enumerate(batches):
        with torch.no_grad():
            expected = wrapper(ids).numpy()
        got = session.run(["logits"], {"ids": ids.numpy()})[0]
        if i == 0:
            print(f"  logits shape {list(got.shape)} from a {tuple(ids.shape)} prefix")
            assert got.shape[-1] == VOCAB_SIZE, f"expected {VOCAB_SIZE} logits, got {got.shape[-1]}"
        worst = max(worst, float(np.abs(got - expected).max()))
    return worst


def quantize(source: Path, target: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)


def build_batches(lengths: list[int]) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [
        torch.randint(0, 256, (1, length), dtype=torch.long, generator=generator) for length in lengths
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=str(ROOT / "web" / "weights" / "model.onnx"))
    ap.add_argument("--int8", action="store_true", help="also write a dynamically quantized model.int8.onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.name[: -len(".int8.onnx")] if out.name.endswith(".int8.onnx") else out.stem
    fp32 = out.parent / f"{stem}.onnx"
    int8 = out.parent / f"{stem}.int8.onnx"

    model, ckpt = load_model(checkpoint)
    wrapper = LastPositionLogits(model).eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"checkpoint {checkpoint}  step={ckpt.get('step')}  params={params}")

    export_fp32(wrapper, fp32, args.opset)
    print(f"wrote {fp32}  ({fp32.stat().st_size / 1e6:.2f} MB)")
    print(f"  graph {describe_graph(fp32)}")

    batches = build_batches([1, 9, 40, 64])
    diff = compare(fp32, wrapper, batches)
    print(f"  fp32 max abs diff vs torch: {diff:.3e}")

    if args.int8:
        quantize(fp32, int8)
        print(f"wrote {int8}  ({int8.stat().st_size / 1e6:.2f} MB)")
        print(f"  graph {describe_graph(int8)}")
        int8_diff = compare(int8, wrapper, batches)
        ratio = fp32.stat().st_size / int8.stat().st_size
        print(f"  int8 max abs diff vs torch: {int8_diff:.3e}  ({ratio:.2f}x smaller)")


if __name__ == "__main__":
    main()

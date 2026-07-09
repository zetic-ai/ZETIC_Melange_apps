#!/usr/bin/env python3
"""
export.py — RetinaDRScreen (Stage-0 re-runnable export recipe).

Family: Medical-imaging CLASSIFICATION (color fundus / retinal DR screening).
        Recipe = transformers image-classifier -> torch.onnx.export, STATIC input
        [1,3,224,224], opset 12, legacy TorchScript exporter (dynamo=False),
        do_constant_folding=True, half=False (Melange owns precision), NO dynamic axes.

Winner model: EscvNcl/MobileNet-V2-Retinopathy
  arch:   transformers MobileNetV2ForImageClassification (MobileNetV2-1.4 backbone),
          a BINARY head. NOT a 5-grade severity classifier.
  task:   REFERABLE-DR screening — NRDR (not-referable) vs RDR (referable, DR grade >= 2).
  labels: id2label = {0: "Nrdr", 1: "Rdr"}  ->  referable = index 1.
  size:   ~17 MB ONNX (smallest of the 6-way bakeoff — see model_selection.md).
  license: `other` (UNDECLARED terms) — pre-ship legal check, flagged in the docs.

Why this recipe / model (full shortlist in model_selection.md):
  MobileNetV2 is a standard, mobile-first CNN — exported at opset 12 the whole graph
  is ordinary ops (Conv/Add/Clip/Relu6/GlobalAveragePool/Gemm), no dynamic axes, no
  attention. Ideal Melange fit and by far the smallest artifact (~17 MB) versus the
  ViT / Swin / EfficientNet DR candidates (112-470 MB) which fight Melange's compile
  step and carry the ViT/MPSGraph GPU-crash risk. This app is the compact on-device
  REFERABLE screener; the ViT 5-grade severity option lives in the sibling app
  RetinaDRGrade.

What this script does, re-runnably:
  1. Download the winner from the Hugging Face Hub (transformers auto-loads it).
  2. Wrap it so forward(pixel_values) returns RAW LOGITS float32[1,2].
  3. torch.onnx.export with STATIC input [1,3,224,224], opset 12, dynamo=False,
     do_constant_folding=True, half=False, NO dynamic axes.
  4. onnx.checker + read back the ACTUAL input/output shapes and op set.
  5. torch-vs-onnxruntime parity check on a random tensor.
  (sample_input.npy is generated separately: np.random.rand(1,3,224,224).float32 —
   Melange only needs shape+dtype; it does NOT encode the preprocessing below.)

IMPORTANT — preprocessing the model expects (from preprocessor_config.json;
  MobileNetV2ImageProcessor). The Dart pipeline must reproduce this EXACTLY — it is
  NOT a plain /255 like the YOLO apps:
    1. Resize so the SHORTEST edge = 256 (bilinear, PIL resample=2), preserving aspect.
    2. Center-crop 224 x 224.
    3. float32, rescale * 1/255                 -> [0,1].
    4. Normalize per channel (v - 0.5) / 0.5    -> [-1,1]  (mean=std=[0.5,0.5,0.5]).
    5. HWC -> NCHW, add batch -> [1,3,224,224], RGB channel order.

Output semantics:
  ONNX output `logits` = float32[1,2] RAW LOGITS (unnormalized).
  Downstream (Dart): softmax over the 2 logits; P(referable) = softmax[index 1].
  Decision: referable if P(index 1) >= threshold (default 0.5; the app may expose it).

Setup:
  uv venv --python 3.12 venv && source venv/bin/activate
  uv pip install torch transformers onnx onnxruntime huggingface_hub numpy
  python export.py
"""
import os
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification
from huggingface_hub import snapshot_download

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
HF_REPO = "EscvNcl/MobileNet-V2-Retinopathy"
IMG = 224
OPSET = 12
ONNX_NAME = "mobilenetv2-dr-referable.onnx"
LABELS = {0: "Nrdr", 1: "Rdr"}  # referable = index 1


class Wrap(nn.Module):
    """Expose RAW logits from a plain pixel_values tensor (no HF output struct)."""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(pixel_values=x).logits


def main() -> None:
    path = snapshot_download(HF_REPO)
    model = AutoModelForImageClassification.from_pretrained(path).eval()
    print(f"[load] {HF_REPO}  id2label={model.config.id2label} (referable = index 1)")

    wrapped = Wrap(model).eval()
    dummy = torch.randn(1, 3, IMG, IMG)  # STATIC shape

    dst = HERE / ONNX_NAME
    torch.onnx.export(
        wrapped,
        dummy,
        str(dst),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=OPSET,
        do_constant_folding=True,
        dynamic_axes=None,   # STATIC — no dynamic axes
        dynamo=False,        # legacy TorchScript exporter -> clean static graph
    )
    print(f"[onnx] -> {dst}")

    m = onnx.load(str(dst))
    onnx.checker.check_model(m)
    ops = sorted(set(n.op_type for n in m.graph.node))

    def dims(t):
        return [d.dim_value if d.dim_value else (d.dim_param or "?")
                for d in t.type.tensor_type.shape.dim]

    print("onnx.checker: PASS ; size(MB)=%.2f" % (os.path.getsize(dst) / 1e6))
    print("\n== ACTUAL ONNX I/O ==")
    for i in m.graph.input:
        print(f"  input  {i.name}: {dims(i)}")
    for o in m.graph.output:
        print(f"  output {o.name}: {dims(o)}")
    print(f"  opset: {m.opset_import[0].version}  op types: {ops}")
    dynamic = any(not d.dim_value for t in list(m.graph.input) + list(m.graph.output)
                  for d in t.type.tensor_type.shape.dim)
    print(f"  dynamic axes present? {dynamic}  (must be False)")

    # torch-vs-onnxruntime parity
    x = np.random.rand(1, 3, IMG, IMG).astype(np.float32)
    with torch.no_grad():
        ty = wrapped(torch.from_numpy(x)).numpy()
    oy = ort.InferenceSession(str(dst)).run(None, {"pixel_values": x})[0]
    print(f"\n[parity] max|torch-onnx| = {np.abs(ty - oy).max():.3e}")
    print(f"[labels] {LABELS}  (output is RAW LOGITS float32[1,2] -> softmax in Dart; "
          f"referable = index 1)")


if __name__ == "__main__":
    main()

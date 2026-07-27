#!/usr/bin/env python3
"""
export.py — ContentModeration (Stage-0 re-runnable export recipe).

Family: Image CLASSIFICATION (on-device content-safety / NSFW screening).
        Recipe = timm ViT image-classifier -> torch.onnx.export, STATIC input
        [1,3,384,384], opset 12, legacy TorchScript exporter (dynamo=False),
        do_constant_folding=True, half=False (Melange owns precision), NO dynamic axes.

Winner model: Marqo/nsfw-image-detection-384
  arch:    timm `vit_tiny_patch16_384` (ViT-Tiny, patch16, 384px), 5.6M params.
  task:    BINARY image content-safety — NSFW vs SFW. Purpose-built by Marqo for
           on-device / at-scale image moderation; their published eval claims it
           beats Falconsai/nsfw_image_detection on their benchmark.
  labels:  config label_names = ["NSFW", "SFW"]  ->  index 0 = NSFW, index 1 = SFW.
  size:    ~22 MB safetensors -> ~22 MB fp32 ONNX. By FAR the smallest credible
           NSFW model on HF (the ViT-base alternatives are 343-344 MB).
  license: Apache-2.0 (commercial/demo use OK).

Why this model / recipe (full shortlist + measured numbers in model_selection.md):
  Validated head-to-head against Falconsai/nsfw_image_detection (343 MB),
  AdamCodd/vit-base-nsfw-detector (344 MB) and viddexa/nsfw-detection-2-mini
  (70 MB, 5-class) on a 36-image ground-truth-labeled safe/borderline test set.
  Marqo: 100% SFW specificity, lowest worst-case false-positive on safe content
  (max P(NSFW)=0.080), and a clean MONOTONIC score gradient
  (safe 0.055 -> art 0.111 -> swimwear 0.236) that is ideal for a keep/blur/block
  score-band demo. Falconsai matched specificity but its scores are saturated to
  ~0.000 on everything short of explicit content (no gradient to demo) AND it is
  15x the size. viddexa was rejected (flagged a legal-handgun photo at P=1.000).
  vit_tiny is a standard timm ViT: exported at opset 12 the graph is ordinary ops
  (Conv patch-embed, MatMul/Add/Softmax attention, LayerNorm, Gemm), STATIC shapes,
  no dynamic axes, no control flow. (Attention heads carry the known iOS-26 MPSGraph
  *GPU* crash risk — that is an OS bug handled server-side by ZETIC, NOT a selection
  criterion; see CLAUDE.md section 5.)

HONEST TASK-FIT CAVEAT (see model_selection.md): this is the NSFW/sexual-content
  axis ONLY. It does NOT classify violence, gore or weapons as separate categories
  (a legal weapon photo is correctly SFW to this model). No commercially-licensed,
  mobile-sized, well-validated MULTI-LABEL content-safety model (NSFW+violence+
  weapons+gore in one head) exists on HF at demo quality; a reliable binary NSFW gate
  is the credible Trust & Safety core. keep/blur/block is driven by score bands on the
  single P(NSFW).

What this script does, re-runnably:
  1. Load the winner from the Hugging Face Hub via timm (hf-hub: prefix).
  2. Wrap it so forward(pixel_values) returns RAW LOGITS float32[1,2].
  3. torch.onnx.export with STATIC input [1,3,384,384], opset 12, dynamo=False,
     do_constant_folding=True, half=False, NO dynamic axes.
  4. onnx.checker + read back the ACTUAL input/output shapes and op set.
  5. torch-vs-onnxruntime parity check on a random tensor.
  6. Write sample_input.npy = np.random.rand(1,3,384,384).float32 (Melange only
     needs shape+dtype; it does NOT encode the preprocessing below).

IMPORTANT — preprocessing the model expects (from the timm pretrained_cfg;
  vit_tiny_patch16_384 augreg). The Dart pipeline must reproduce this EXACTLY — it is
  NOT a plain /255 like the YOLO apps:
    1. Resize so the SHORTEST edge = 384 (BICUBIC, antialias=True), preserving aspect.
    2. Center-crop 384 x 384.
    3. float32, rescale * 1/255                 -> [0,1].
    4. Normalize per channel (v - 0.5) / 0.5    -> [-1,1]  (mean=std=[0.5,0.5,0.5]).
    5. HWC -> NCHW, add batch -> [1,3,384,384], RGB channel order.

Output semantics:
  ONNX output `logits` = float32[1,2] RAW LOGITS (unnormalized), order [NSFW, SFW].
  Downstream (Dart): softmax over the 2 logits; P(NSFW) = softmax[index 0].
  Decision bands (default, app may expose thresholds):
    P(NSFW) <  0.30           -> KEEP
    0.30 <= P(NSFW) < 0.70    -> BLUR / review
    P(NSFW) >= 0.70           -> BLOCK

Setup:
  # reuses ZETIC's melange-env (torch/onnx/onnxruntime/huggingface_hub) + timm:
  #   pip install timm safetensors
  python export.py
"""
import os
import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
from timm.layers import set_fused_attn

warnings.filterwarnings("ignore")

# Export the attention as EXPLICIT matmul + softmax + matmul rather than a single
# fused aten::scaled_dot_product_attention node. SDPA is not exportable at opset 12
# (needs 14+), and the explicit form is the more Melange-friendly graph anyway:
# only standard ops (MatMul / Softmax / Add / Mul), no fused attention node.
# Numerically equivalent (verified by the torch-vs-onnxruntime parity check below).
set_fused_attn(False)

HERE = Path(__file__).resolve().parent
HF_REPO = "Marqo/nsfw-image-detection-384"
IMG = 384
OPSET = 12
ONNX_NAME = "nsfw-vit-tiny-384.onnx"
LABELS = {0: "NSFW", 1: "SFW"}  # config label_names order; NSFW = index 0


class Wrap(nn.Module):
    """Expose RAW logits from a plain pixel_values tensor."""
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)


def main() -> None:
    model = timm.create_model(f"hf-hub:{HF_REPO}", pretrained=True).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[load] {HF_REPO}  params={n_params:.2f}M  labels={LABELS} (NSFW = index 0)")

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

    # sample_input.npy — shape+dtype only (random noise is fine for Melange)
    np.save(HERE / "sample_input.npy", x)
    print(f"[sample] sample_input.npy  shape={x.shape} dtype={x.dtype}")
    print(f"[labels] {LABELS}  (output is RAW LOGITS float32[1,2] -> softmax in Dart; "
          f"P(NSFW) = softmax[index 0])")


if __name__ == "__main__":
    main()

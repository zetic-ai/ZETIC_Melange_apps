# Model selection — ContentModeration (image classification, use-case: on-device content-safety / NSFW screening)

Target sector: Trust & Safety (~19 target companies: Bumble, Match, Grindr, Discord,
Reddit, OfferUp, Snap, WebPurify, Sightengine, Hive, Clarifai, …). Demo story:
moderate a user image / video frame fully on-device before upload — no cloud call —
and drive a keep / blur / block decision.

## Shortlist (top 5)

| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 | **Marqo/nsfw-image-detection-384** | 172K | apache-2.0 | timm ViT -> torch.onnx (opset 12, static) | **22 MB**, `vit_tiny_patch16_384`, 5.6M params. Binary NSFW/SFW. Only credible NSFW model in the mobile-size sweet spot. Clean static ONNX, standard ops. Purpose-built; Marqo's published eval claims it beats Falconsai. | **9.3** |
| 2 | Falconsai/nsfw_image_detection | 8.7M | apache-2.0 | transformers ViT -> onnx | Gold-standard binary NSFW (normal/nsfw), ViT-base. But **343 MB** (10x the largest committed onnx in-repo; fights Melange compile) and scores are SATURATED (~0.000 on everything short of explicit) so it shows NO gradient for a keep/blur/block demo. | 7.4 |
| 3 | AdamCodd/vit-base-nsfw-detector | 652K | apache-2.0 | ships onnx/ + transformers ViT | Binary sfw/nsfw, ViT-base **384px, 344 MB**. Good score gradient (best separation among binaries) but same prohibitive size. Pre-built ONNX exports exist (incl. quantized). | 7.6 |
| 4 | Freepik/nsfw_image_detector | 466K | mit | TimmWrapper -> onnx | Multi-severity NSFW (timm wrapper), **172 MB**. Richer taxonomy but big and its config carries no clean id2label; less-documented I/O. | 6.5 |
| 5 | viddexa/nsfw-detection-2-mini | 753 | apache-2.0 | transformers EfficientNet -> onnx | 5-class (safe/hentai/porn/sexy/drawing), EfficientNet **70 MB**, 380px. Attractive taxonomy for keep/blur/block BUT only 753 downloads (weak signal) and **REJECTED in validation** (flagged a legal-handgun photo at P(NSFW)=1.000 — a demo-killer false positive). | 5.2 |

Also considered and rejected up front:
- **giacomoarienti/nsfw-classifier** (223K dl, 5-class porn/hentai/sexy/drawings/neutral): license **cc-by-nc-nd-4.0 — NON-COMMERCIAL**, disqualified for a GTM demo.
- **prithivMLmods/siglip2-x256-explicit-content** (94K dl, 5-class explicit taxonomy): SigLIP2, **372 MB**, exotic/large; poor Melange fit.
- Violence / weapon axes: **jaranohaal/vit-base-violence-detection** (undocumented labels, low signal) and only a toy CS:GO weapon classifier exist. No demo-quality, commercially-licensed, mobile-sized model.

## Winner: Marqo/nsfw-image-detection-384

Why this one over the runners-up (Melange-fit + task-fit trade-off):
- **Size is decisive.** At 22 MB it is the *only* credible NSFW model in the rubric's
  "single-to-low-tens MB" band and fits the repo's committed-onnx convention (largest
  existing is 36 MB). The ViT-base alternatives (Falconsai 343 MB, AdamCodd 344 MB,
  Freepik 172 MB, SigLIP2 372 MB) are 8–17x larger and fight Melange's compile step.
- **Best measured demo behavior.** On a 36-image ground-truth-labeled safe/borderline
  test set (below) it hit **100% SFW specificity**, the **lowest worst-case false
  positive on safe content** (max P(NSFW)=0.081), and a clean **monotonic score
  gradient** (safe 0.054 -> classical-art 0.165 -> swimwear 0.229) — exactly what a
  keep/blur/block score-band demo needs. Falconsai matched specificity but its scores
  collapse to ~0.000 (no gradient to show); viddexa was rejected for a P=1.000 false
  positive on a legal weapon photo.
- **Clean license + clean export.** Apache-2.0. Exports to a static opset-12 ONNX of
  ordinary ops (Conv/MatMul/Softmax/LayerNorm/Gemm) with fused attention disabled — no
  SDPA node, no dynamic axes.

### HONEST task-fit caveat (loud, per EXPLORATION.md §7)
This is the **NSFW / sexual-content axis ONLY** — a **binary** NSFW vs SFW gate. It
does **not** classify violence, gore or weapons as separate categories (it correctly
treats a legal weapon photo as SFW, P(NSFW)≈0.05). The brief's ideal (one multi-label
head over NSFW+violence+weapons+gore) **does not exist on HF at demo quality**: the
multi-label options are either non-commercial (giacomoarienti), huge/exotic (SigLIP2),
or weak/unvalidated (viddexa — and it fails on safe content). Per the assignment's own
guidance, **a reliable smaller taxonomy beats a flaky big one for a live demo**, so the
credible Trust & Safety core is a rock-solid binary NSFW gate. keep/blur/block is driven
by score bands on the single P(NSFW):
`KEEP < 0.30 <= REVIEW/BLUR < 0.70 <= BLOCK` (defaults; app may expose thresholds).

## Validation (measured — VALIDATION-GATED SELECTION, EXPLORATION.md §7)

Ground-truth test set: **36 images** assembled from Wikimedia Commons — 28 clearly-safe
(person 8, animal 6, food 4, legal-weapon 4, scene 3, object 3) + 8 borderline NSFW-axis
proxies (swimwear 4, classical nude art 4). Per the assignment, **no explicit imagery
was collected**; borderline safe-to-display proxies test score separation and the
review/block bands, and true explicit-content recall is taken from Marqo's published
eval. Metric of record: **SFW specificity** (fraction of safe images the model keeps,
P(NSFW)<0.5) — the false-positive-on-safe-content rate is what makes or breaks a live
demo — plus mean-score separation between safe and borderline.

Head-to-head (torch, each model's exact preprocessing):

| Model | Size | SFW specificity | max FP on safe | mean P(NSFW) safe | borderline | separation | verdict |
|-------|------|-----------------|----------------|-------------------|------------|------------|---------|
| **Marqo/nsfw-image-detection-384** | 22 MB | **100% (28/28)** | **0.080** | 0.055 | 0.173 | +0.119 | **WINNER** |
| Falconsai/nsfw_image_detection | 343 MB | 100% (28/28) | 0.002 | 0.000 | 0.000 | +0.000 | too big; no gradient |
| AdamCodd/vit-base-nsfw-detector | 344 MB | 100% (28/28) | 0.133 | 0.024 | 0.167 | +0.142 | too big |
| viddexa/nsfw-detection-2-mini | 70 MB | 96.4% (27/28) | **1.000** | 0.037 | 0.335 | +0.298 | REJECTED (FP=1.0 on safe weapon) |

Winner re-confirmed on the **actual exported ONNX** (onnxruntime + the exact app
preprocessing pipeline, not the timm transform): 100% SFW specificity, max FP on safe
0.081, mean P(NSFW) safe 0.054 -> borderline 0.197; ONNX-vs-torch mean |ΔP(NSFW)| =
0.0073 across all 36 images (single-image max Δ 0.22 on one high-frequency borderline
image, attributable to resize/antialias differences — safe class unaffected). Per-category
safe means all in 0.049–0.059. See `demo_images/results.json` for the curated demo triplet
spanning KEEP / REVIEW-BLUR / BLOCK, all non-explicit.

## Export
- Recipe: `export.py` (timm ViT family; `set_fused_attn(False)` then torch.onnx.export).
- Input:  float32[1,3,384,384], NCHW, RGB. Preprocessing: resize shortest-edge 384
  (BICUBIC, antialias) -> center-crop 384 -> /255 -> (x-0.5)/0.5 (mean=std=0.5) -> [-1,1].
- Output: float32[1,2] RAW LOGITS, order [NSFW, SFW]. Post-processing: softmax in Dart;
  P(NSFW)=softmax[0]. NOT baked into ONNX. No NMS / no activation baked in.
- Opset 12, static shapes confirmed (dynamic axes present? False). Fused attention
  disabled -> no scaled_dot_product_attention node; all standard ops. onnx.checker PASS.
  torch-vs-onnxruntime parity max|Δ| = 3.4e-07. ONNX size 22.5 MB.

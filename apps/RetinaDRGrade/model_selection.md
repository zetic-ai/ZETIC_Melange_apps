# Model selection — RetinaDRGrade (diabetic-retinopathy grading, use-case: 5-grade SEVERITY grader)

A 6-way bakeoff was run over candidate DR classifiers under
`../RetinaDRScreen/_eval/*/`, all scored on the SAME 42-image held-out set (IDRiD + APTOS
fundus images; GT distribution grades 0..4 = 6/12/12/6/6). This app carries the richer
5-grade SEVERITY grader that won on accuracy. The sibling app **RetinaDRScreen** carries
the tiny binary MobileNetV2 screener (below) for low-end / offline device-fit.

## Shortlist / bakeoff (6 models, same 42-image eval)
| Rank | HF repo | Arch | Task | Exact-5-grade acc | Referable(>=2) sens / spec | Spans all 5 grades? | License | Notes |
|------|---------|------|------|-------------------|-----------------------------|---------------------|---------|-------|
| 1 | Kontawat/vit-diabetic-retinopathy-classification | ViT-base | 5-class grade | **0.667** (28/42) | **1.00 / 0.833** | yes (non-degenerate) | apache-2.0 | WINNER. Best exact acc, perfect referable sensitivity, clean per-grade spread. ~343 MB fp32. |
| 2 | Augusto777/... SwinV2-tiny | SwinV2-tiny | 5-class grade | 0.524 (22/42) | 1.00 / 0.667 | no — never predicts grade 3; 29/42 collapse to grade 2 | (permissive) | Good referable sens but degenerate mid-grades; can't tell Severe from Moderate. |
| 3 | vyshnav/... EfficientNet-B5 | EfficientNet-B5 | ordinal regression (score->round) | 0.452 (19/42) | 0.917 / 0.611 | yes | (permissive) | Regression head; weaker exact acc and referable specificity. |
| 4 | rafalosa/diabetic-retinopathy-... (procnorm ViT) | ViT | 5-class grade | 0.381 (16/42) | 0.875 / 0.722 | no — only ever predicts No DR + Moderate | (permissive) | Heavily degenerate; unusable as a true grader. |
| 5 | karthick/... EfficientNetV2-L | EfficientNetV2-L | ordinal regression (score->round) | 0.119 (5/42) | 1.00 / 0.056 | no — collapses to grades 3/4 | (permissive) | Over-predicts referable for everything; specificity ~0. Large model, poor fit. |
| — | EscvNcl/MobileNet-V2-Retinopathy | MobileNetV2 | BINARY (Nrdr/Rdr) | n/a (2-class only) | 0.833 / 0.889 (binary acc 0.857) | n/a | (permissive) | Not a 5-grade grader — it is the binary referable screener. Tiny + fast; assigned to sibling app RetinaDRScreen for offline/low-end device-fit. |

(Exact acc and referable sens/spec recomputed from each dir's predictions/summary. Only
Kontawat ViT-base and EfficientNet-B5 span all five grades non-degenerately; of those the
ViT is clearly stronger on every metric.)

## Winner: Kontawat/vit-diabetic-retinopathy-classification
Why this one over the runners-up (Melange-fit + task-fit trade-off):
- **Best exact-grade accuracy (0.667)** and the only model with **perfect referable
  sensitivity (1.00) AND high specificity (0.833)** — for a severity grader you want both
  "never miss a sight-threatening eye" and "don't over-refer healthy ones."
- **Uses all five grades non-degenerately** — the Swin and both EfficientNet regressors
  collapse toward a single mode (grade 2, or 3/4), so they can't actually grade severity;
  the ViT gives a real per-grade confidence distribution (see demo viz).
- **Clean, standard ViT-base** — conventional attention exports cleanly to ONNX; identity
  `id2label` means argmax == canonical grade with zero remapping. **apache-2.0** (clean for
  ZETIC GTM).
- The MobileNetV2 is more device-friendly but is only a **binary** referable screener — it
  cannot produce a 0..4 severity grade, which is this app's entire purpose. It lives in the
  sibling **RetinaDRScreen** for offline/low-end fit.

Honest trade-off: the ViT-base ONNX is **~343 MB fp32** — heavy for on-device (a
first-launch download + storage cost). That is the price of the accuracy + full-grade
capability; the binary screener sibling covers the small/offline end of the spectrum.

## Export
- Recipe: `export.py` (transformers ViTForImageClassification, `attn_implementation="eager"`
  for opset-12 compatibility, `torch.onnx.export(dynamo=False)`, static shapes, half=False).
- Input:  float32[1,3,224,224], NCHW RGB; resize 224 bilinear, /255, normalize
  mean/std [0.5,0.5,0.5] (pixels -> [-1,1]).
- Output: float32[1,5] RAW logits; no softmax/argmax baked in (downstream softmax+argmax).
- Opset 12, static shapes confirmed, onnx.checker PASS; torch-vs-onnxruntime parity verified.

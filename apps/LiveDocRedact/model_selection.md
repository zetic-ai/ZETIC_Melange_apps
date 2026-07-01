# Model selection — LiveDocRedact (OCR, use-case: live sensitive-document capture + PII auto-redaction)

Sector: **fintech ID-scanner + healthcare** (ZETIC's strongest on-device privacy story —
ID / passport / medical forms must never be streamed to a cloud OCR).

This is a **TWO-MODEL pipeline** (orchestrator-decided): a text **DETECTOR** that finds
text-region boxes in the full frame + a text **RECOGNIZER** that reads characters from each
cropped region. The text-region grouping / crop-and-feed-recognizer orchestration is Dart
(worker's job later). Each model registers as its own Melange model.

## Shortlist (candidates for both stages, scored on the Melange-fit rubric)

Score is 0-10 (exportable · static-shape · standard-ops · mobile-size · license · popularity ·
task-fit · known-I/O). **Both winners are Apache-2.0 — GTM-clean, no license gate to resolve.**

| Rank | Stage | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|-------|---------|-----------|---------|-------------|-------------------|-------|
| 1 — **WINNER (det)** | DET | **PaddlePaddle/PP-OCRv5_mobile_det_onnx** (`inference.onnx`) | 134,282 | **Apache-2.0** | **Pre-exported ONNX**; pin input `[N,3,H,W]`→`[1,3,640,640]` + onnxslim-fold | DBNet (Conv/BN/ConvTranspose/Sigmoid/Resize; opset 11→12). Fully-conv → clean static. **4.75 MB.** No paddle runtime needed. Output = 1-channel probability heatmap; DB post-proc is pure-Dart. | **9.2** |
| 1 — **WINNER (rec)** | REC | **PaddlePaddle/en_PP-OCRv5_mobile_rec** (`inference.json`+`.pdiparams`) | 294,731 | **Apache-2.0** | paddle2onnx (opset 12) → pin `[N,3,48,W]`→`[1,3,48,320]` + onnxslim-fold | CRNN/SVTR CTC head. **English/Latin charset = lean 438-class head** ([1,40,438]) vs 18,385 multilingual — far cheaper per-crop decode for Latin IDs. **7.8 MB.** Charset shipped as `en_dict.txt`. | **9.0** |
| 2 | REC | PaddlePaddle/PP-OCRv5_mobile_rec_onnx (`inference.onnx`) | 79,885 | Apache-2.0 | Pre-exported ONNX; pin width→320 + fold (no paddle) | Same architecture, **pre-exported (no paddle2onnx step)** → the lowest-friction fallback. But **18,385-class Chinese head** → [1,40,18385] output = ~46× heavier CTC argmax per crop (bad for many-region live frames) and a giant charset. Kept as drop-in fallback if the paddle export is ever unavailable. | 7.9 |
| 3 | DET | PaddlePaddle/PP-OCRv6_medium_det_onnx | 74,645 | Apache-2.0 | Pre-exported ONNX | Newer v6, but **"medium"** (heavier than mobile) and the `PP-OCRv6_mobile_det_onnx` repo 404/401s (not public). v5-mobile is the better-proven on-device size. | 7.4 |
| 4 | REC | PaddlePaddle/latin_PP-OCRv5_mobile_rec | 38,905 | Apache-2.0 | paddle2onnx → pin + fold | Latin-multilingual (accented EU chars — good for EU IDs), but larger charset than English-only and no extra popularity/eval edge for a US-first demo. Solid EU-locale alternative (swap `REC_REPO` in `export.py`). | 8.3 |
| 5 | REC | Felix92/onnxtr-crnn-mobilenet-v3-small | 227 | Apache-2.0 | Pre-exported ONNX (docTR/OnnxTR) | CRNN-MobileNetV3, already ONNX & small. But **docTR's own charset/decoder** (different from PP-OCR), low popularity/eval signal, and its detector half is a separate docTR model — mixing families loses the single-recipe benefit. Rejected on task/ecosystem fit. | 6.8 |
| — | REC | software-mansion/react-native-executorch-recognizer-crnn.en | 16,322 | (RN-ExecuTorch) | ExecuTorch `.pte`, not ONNX | Popular but **ExecuTorch-packaged, not an ONNX export path** → disqualified for Melange. | — |
| — | DET/REC | karlo0/surya_text_recognition · xiaoyao9184/surya_* | ~200 | GPL-family/varies | transformers ONNX | Surya = ViT/transformer OCR: exotic attention, typically **dynamic shapes**, heavier — fights Melange compile (the family the iOS-26 MPSGraph class of bug lives near). Rejected on Melange-fit. | — |

## Winner: PP-OCRv5 mobile **DBNet detector** + **English CRNN/SVTR CTC recognizer**
Why this pair over the runners-up (Melange-fit + task-fit, in 2-4 lines):
- **Both Apache-2.0** → no license gate can sink the GTM demo (the license column is fully clean;
  contrast VehiclePlateYOLO where an AGPL pick had to be demoted).
- **The de-facto on-device OCR pair**: PP-OCR mobile det+rec is the most-deployed lightweight OCR
  stack, ~4.75 MB + ~7.8 MB, standard conv/BN/CTC ops that convert cleanly, and both fold to
  **fully static ONNX** (verified — no dynamic axis, no `Shape` op survives).
- **English recognizer is the deliberate task-fit call**: IDs/passports/medical forms are Latin +
  digits, so the 436-char English head ([1,40,**438**]) beats the 18,385-class multilingual head on
  per-crop decode cost — which matters because a live frame feeds the recognizer once **per detected
  region** (tens of crops/frame). Trade-off accepted: it will not read CJK; for EU accents swap to
  `latin_PP-OCRv5_mobile_rec` (one-line `REC_REPO` change), and the pre-exported multilingual
  `PP-OCRv5_mobile_rec_onnx` is the zero-paddle fallback.

## Export — DETECTOR (`doc_text_detector.onnx`)
- Recipe: `export.py` → HF pre-exported ONNX, input dims pinned to `[1,3,640,640]`, onnxslim-folded.
- Input:  float32 `x` **[1,3,640,640]**, NCHW, **BGR** (PaddleOCR cv2 order — see trap below),
  ImageNet-normalized: `(pixel/255 − mean)/std`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- Output: float32 `fetch_name_0` **[1,1,640,640]** = a single **probability heatmap** in 640×640
  letterboxed space (values ~0..1). **Post-processing NOT baked in** — DB decode (binarize `thresh≈0.3`,
  connected-components/contours, `unclip_ratio≈1.5` dilation, `box_thresh≈0.6` filter → quad boxes) is
  pure-Dart. opset 12 (source ONNX opset 11 upgraded via onnx.version_converter). **Static shapes confirmed** (no dynamic axes, no `Shape` op; ORT run → (1,1,640,640)).

## Export — RECOGNIZER (`doc_text_recognizer.onnx`)
- Recipe: `export.py` → paddle2onnx (opset 12), input pinned to the **FIXED** `[1,3,48,320]`, onnxslim-folded.
- Input:  float32 `x` **[1,3,48,320]**, NCHW, **BGR**, PP-OCR rec norm `(pixel/255 − 0.5)/0.5` → **[-1,1]**.
  **Fixed width = 320** (variable width is a static-shape violation): resize each crop keeping aspect to
  H=48, then right-pad with zeros to W=320 (or downscale if wider). This is the documented static resolution.
- Output: float32 `fetch_name_0` **[1,40,438]** = 40 CTC time-steps × 438 classes (softmax probabilities).
  **CTC classes = [blank](idx 0) + 436 dict chars (idx 1..436) + space ' '(idx 437).** Dict shipped as
  `en_dict.txt` (one char/line; line i → class i+1). **Decode NOT baked in** — greedy CTC (per-step argmax,
  collapse repeats, drop blank) is pure-Dart. opset 12. **Static shapes confirmed** (no dynamic axes, no
  `Shape` op; ORT run → (1,40,438)).

## Charset / dictionary (recognizer) — REQUIRED Dart asset
`en_dict.txt` (436 entries) is emitted by `export.py` from the model's own `inference.yml`. The Dart
CTC decoder MUST build its label list as **`['blank'] + en_dict.txt lines + [' ']`** (blank at 0, space at
437) to match the 438-wide head. Getting this list wrong yields plausible-but-wrong text — cover it in Tier A.

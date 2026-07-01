# Model selection — ReceiptScanner (OCR, use-case: receipt / invoice text extraction)

Sector: field-sales / enterprise expense. On-device justification: expense capture runs
in poor-connectivity places and receipts carry financial data an enterprise may not want
in a third-party cloud OCR — so the whole OCR pipeline must run offline on-device.

Family: OCR (PaddleOCR PP-OCRv5 mobile). First OCR-family run → this also establishes the
reusable PP-OCR export recipe (`export.py`: paddle2onnx → onnxslim static-shape fix).

## Shortlist (top 5)

Scored on the Melange-fit rubric (ONNX-exportable, static-shape, standard ops, mobile
size, license, popularity, task fit, known output). "Score" is the Melange+task
technical ranking; the winner set is decided by that plus the receipt/GTM fit.

| Rank | HF repo | Downloads | License | Export path | Melange-fit notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 — **SELECTED (rec)** | **PaddlePaddle/en_PP-OCRv5_mobile_rec** | 294,731 | **Apache-2.0** | paddle2onnx → onnxslim (opset 12) | English SVTR/CTC recognizer. 436-char **receipt-friendly** charset (digits, A–Z/a–z, punctuation, currency `$ ¢ ₤ ₹ ₽ €`-adjacent, `%`, checkmarks). Static [1,3,48,320]→[1,40,438]. ~7.5 MB. Trivial 438-class CTC decode. Std conv/matmul/softmax ops. **Verified static + opset 12.** | 9.2 |
| 1 — **SELECTED (det)** | **PaddlePaddle/PP-OCRv5_mobile_det** | 134,282 | **Apache-2.0** | paddle2onnx → onnxslim (opset 12) | DB (Differentiable Binarization) text detector. Fully-conv (Conv/BN/Resize/ConvTranspose/Sigmoid) → clean Melange fit. Static [1,3,960,640]→[1,1,960,640] prob map. ~4.5 MB. Localizes the many text lines on a receipt (a recognizer can't). **Verified static + opset 12.** | 9.0 |
| 2 — alternative (rec) | PaddlePaddle/PP-OCRv5_mobile_rec (multilingual) | 79,885 | Apache-2.0 | ready ONNX (opset 7) → onnxslim | Ready pre-exported ONNX, no paddle2onnx step. But **18,385-class CJK-heavy** head → 16 MB, unwieldy Dart charset, larger output tensor. Overkill for English/Latin receipts. Drop-in fallback if non-Latin receipts are ever needed. | 8.0 |
| 3 | PaddlePaddle/PP-OCRv6_medium_{det,rec}_onnx | 74k / 68k | Apache-2.0 | ready ONNX | Newer (2025) PP-OCRv6, ready ONNX. Higher accuracy but "medium" = larger/slower than v5 *mobile*; v6 is newer/less field-proven. Good future upgrade once v5 path is validated on-device. | 7.8 |
| 4 | microsoft/trocr-base-printed | 246,093 | none declared | transformers → ONNX | TrOCR (ViT encoder + autoregressive BART decoder). **Rejected on Melange-fit:** autoregressive generation = dynamic sequence length + a per-token decode loop (not a single static forward pass); ~govern by exotic attention; ~330 MB; **no declared license**. Fundamentally not a static single-pass Melange model. | 3.5 |
| 5 | naver-clova-ix/donut-base | 177,128 | MIT | transformers → ONNX | Donut = OCR-free doc-understanding VisionEncoderDecoder (Swin + BART). Elegant "image→JSON" but **autoregressive + dynamic seq + ~200 MB** → same static-shape / single-pass disqualifier as TrOCR. MIT license is nice; Melange-fit is not. | 3.0 |

## Pipeline decision: TWO models (det + rec), not one — justified

The orchestrator's default is a **single recognition model**. I evaluated that honestly and
it does **not** hold for a full receipt, for a concrete architectural reason:

- A **recognizer** (SVTR/CRNN/TrOCR) consumes **one pre-cropped text line** and emits its
  string. It has **no localization head** — it cannot find, separate, or box the dozens of
  text lines on a receipt. Feeding a whole multi-line receipt into a line-recognizer yields
  garbage (it assumes a single horizontal text strip of height 48).
- The app's contract is **"text + boxes"** (the worker structures line-items/totals/
  merchant/date from those). **Boxes come only from a detector.** So one recognizer can give
  text-without-boxes-for-one-line; one detector can give boxes-without-text. Neither alone
  satisfies the contract for a real receipt.
- A single *end-to-end* model that does both (Donut, TrOCR, PaddleOCR-VL) exists, but every
  such model is **autoregressive + dynamic-shape + hundreds of MB** → the exact profile the
  Melange-fit rubric disqualifies (no static single forward pass). Choosing one to preserve
  "single model" would trade a clean static ONNX for an un-convertible one. That is the wrong
  trade.

Therefore the **justified fallback** is the classic, Melange-clean **detect → recognize**
chain: DB detector produces line boxes; the English SVTR/CTC recognizer reads each cropped
box. Both are small, static, standard-op ONNX. Two Melange models, two forward-pass shapes,
low upload load (~12 MB total). This is exactly the "any model + real post-processing" proof
point: the models emit **text + boxes**; the **Dart layer** does the structured extraction.

### Model's job vs Dart's job (the output contract)
- **DET model →** one probability map `[1,1,960,640]`. Dart runs DB post-processing
  (threshold → connected components/contours → min-area boxes → unclip) to get **N oriented
  line boxes** in the 960×640 letterboxed space, then inverts the letterbox to image space.
- **REC model →** per cropped+deskewed line (resized to 48×320), a CTC matrix `[1,40,438]`.
  Dart greedy-CTC-decodes (argmax per step → collapse repeats → drop blank) to a **string +
  mean confidence** using `rec_charset.json`.
- **Dart structuring (worker, NOT a model):** cluster boxes into rows by y-overlap; sort
  L→R; regex + layout heuristics over the row strings to pull **line-items** (description +
  trailing price token `\d+[.,]\d{2}`), **total** (row matching `TOTAL|AMOUNT DUE|BALANCE`
  with the largest/last currency value), **merchant** (top-of-receipt large text block),
  and **date** (regex over common date formats). This is post-processing, not ML.

## Winner rationale (over the runners-up)
- **English recognizer over multilingual:** the receipt use-case is English/Latin. The
  `en_PP-OCRv5_mobile_rec` charset is 436 receipt-relevant glyphs incl. currency symbols —
  a 438-class CTC head that decodes in a few lines of Dart, vs the multilingual model's
  18,385 CJK-heavy classes (16 MB, unwieldy). Same PP-OCRv5 family, same accuracy on Latin.
- **PP-OCRv5 mobile over v6 / server:** smallest, most field-proven mobile pair; fully-conv
  DB detector converts cleanly; v6/server are the documented upgrade path once v5 is proven
  on hardware.
- **PaddleOCR over TrOCR/Donut:** static single forward pass per model vs autoregressive
  dynamic decode — the decisive Melange-fit gate.

## License
Both models: **Apache-2.0** (PaddlePaddle official repos) — commercially clean for ZETIC's
GTM / trade-show distribution. No restrictive-license flag. (Note the weights were trained
with AGPL-adjacent PaddleOCR *tooling*, but the model weights/repos are published Apache-2.0;
standard for the PaddleOCR model zoo.)

## Export (both models)
- Recipe: `export.py` — `paddle2onnx --opset_version 12` then `onnxslim --input-shapes` to
  pin static dims and constant-fold. (onnxsim segfaulted on arm64; onnxslim is reliable.)
- **DET** `ppocrv5_mobile_det.onnx` (~4.5 MB):
  - Input: float32 `x` **[1,3,960,640]**, NCHW. Preprocess (per PP-OCR `inference.yml`):
    resize/letterbox to 960×640, `/255`, ImageNet-normalize mean `[0.485,0.456,0.406]`
    std `[0.229,0.224,0.225]`, **channel order BGR** (DecodeImage img_mode=BGR), then CHW.
  - Output: float32 `fetch_name_0` **[1,1,960,640]** — DB probability map in `[0,1]`
    (a `Sigmoid` is baked into the graph). **Post-processing NOT baked in** (DB threshold /
    contour / unclip is pure-Dart).
  - Opset 12. **Static shapes confirmed** (checker passes; no dynamic axes in inputs/
    outputs/value_info; onnxruntime forward pass reproduces `[1,1,960,640]`).
- **REC** `en_ppocrv5_mobile_rec.onnx` (~7.5 MB):
  - Input: float32 `x` **[1,3,48,320]**, NCHW. Preprocess: resize each line crop to h=48
    keeping aspect, right-pad to W=320, `/255`, then `(x-0.5)/0.5` → range **[-1,1]**,
    **channel order BGR**, CHW.
  - Output: float32 `fetch_name_0` **[1,40,438]** — CTC posteriors, **softmaxed** (a
    `Softmax` is the final graph op), 40 time-steps × 438 classes. **CTC decode NOT baked
    in** (greedy decode is pure-Dart).
  - **CTC charset:** 438 classes. **blank = index 0**; dict chars = indices 1..436;
    **space = index 437**. Shipped as `rec_charset.json` (438-entry table in index order)
    and `rec_dict.txt` (raw 436-char dict). Sourced from the model's `inference.yml`.
  - Opset 12. **Static shapes confirmed** (as above; forward pass reproduces `[1,40,438]`).

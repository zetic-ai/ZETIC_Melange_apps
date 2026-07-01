# Model selection — SignTranslate (OCR, use-case: LIVE offline scene-text reading for travelers)

Sector: traveler / offline-roaming ("no signal, no roaming data"). Point the camera at a
street sign / menu / label and read the text live as the camera moves. Latency matters
(tracks live); an OPTIONAL local translate step is downstream (Dart, out of scope here —
NO translation model is exported).

**Pipeline shape: TWO models, both required, both registered as separate Melange models.**
Live scene text needs real *detection* of arbitrarily-placed/angled text regions, then a
*recognizer* run per cropped region. Text-region grouping + crop-feed orchestration between
the two models is Dart (worker's job, post GATE 0).

- Detector  →  `ajayshah/SceneTextDetector`   (`ppocrv5_mobile_det.onnx`)
- Recognizer →  `ajayshah/SceneTextRecognizer` (`latin_ppocrv5_mobile_rec.onnx`)

The hard requirement for THIS app: the recognizer must be trained on **scene text** (not
just clean document OCR), or the demo reads garbage on angled signs. That requirement drives
the winner choice below and is defended explicitly.

---

## Shortlist A — text DETECTOR (scene-text)

| Rank | HF repo | Downloads | License | Export path | Melange-fit / scene-text notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 — **SELECTED** | **PaddlePaddle/PP-OCRv5_mobile_det** | 134,282 | **Apache-2.0** | paddle2onnx → onnxslim (static) | DBNet + MobileNetV3, scene+doc trained, arbitrary-orientation text regions. ~4.75 MB ONNX. Static [1,3,736,736]→[1,1,736,736] prob map; opset-12 clean (Conv/HardSigmoid/Sigmoid/Resize/ConvTranspose only). Newest PP-OCR mobile det, strong popularity, clean license. | 9.1 |
| 2 — fallback | PaddlePaddle/PP-OCRv4_mobile_det | 19,426 | Apache-2.0 | same recipe | Prior-gen DBNet mobile; the most battle-tested paddle2onnx det. Drop-in swap (change `DET_REPO` in export.py) if v5 ever misbehaves. Slightly lower recall on hard scenes than v5. | 8.4 |
| 3 | Felix92/onnxtr-db-mobilenet-v3-large | 449 | Apache-2.0 | pre-exported ONNX | DBNet already in ONNX (no conversion), but docTR weights are more document- than scene-oriented, and the graph ships **dynamic** H/W (needs re-fixing). Weaker scene-text signal. | 7.0 |
| 4 | PaddlePaddle/PP-OCRv5_server_det | 600,495 | Apache-2.0 | same recipe | Highest-accuracy DBNet, but server-scale — too heavy for a live per-frame mobile demo. Rejected on mobile-size for a LIVE camera loop. | 6.8 |
| 5 | shuzi-mewtant/dbnet_res18_text_detection_v0.1 | 41 | unclear | mmocr/custom | ResNet18 DBNet; low popularity, thin card, no clean license signal, non-standard export path. Rejected. | 4.5 |

## Shortlist B — text RECOGNIZER (MUST be scene-text)

| Rank | HF repo | Downloads | License | Export path | Melange-fit / scene-text notes | Score |
|------|---------|-----------|---------|-------------|-------------------|-------|
| 1 — **SELECTED** | **PaddlePaddle/latin_PP-OCRv5_mobile_rec** | 38,905 | **Apache-2.0** | paddle2onnx → onnxslim (static) | **SVTR-LCNet, CTC** (NOT autoregressive → clean static opset-12 export). SVTR is a *scene-text* recognition architecture (IC13/IC15/SVT/CUTE); PP-OCR trains it on a large synthetic-scene + real corpus. **Latin charset** = French/Spanish/German/Italian/Portuguese… — the ideal traveler fit. ~8 MB, static [1,3,48,320]→[1,40,838], charset embedded. | 9.3 |
| 2 — alt | PaddlePaddle/en_PP-OCRv5_mobile_rec | 294,731 | Apache-2.0 | same recipe | Same SVTR-LCNet CTC, English-only charset. Most-downloaded, but narrower language coverage than Latin for a traveler. Drop-in swap. | 8.7 |
| 3 — fallback | PaddlePaddle/en_PP-OCRv4_mobile_rec | 25,344 | Apache-2.0 | same recipe | Prior-gen SVTR-LCNet CTC, most-proven paddle2onnx rec. Safe fallback if a v5 quirk appears. | 8.3 |
| 4 — demoted (export risk) | Felix92/onnxtr-parseq-multilingual-v1 | 1,819 | Apache-2.0 | pre-exported ONNX | PARSeq **is** a strong scene-text recognizer, but it is the autoregressive/permuted-attention family the assignment flags as an export trap; the shipped ONNX is dynamic-width and its decode leans on iteration. Higher Melange-compile risk than a CTC head for no decisive scene-text gain. Demoted per the "fall back to CTC" rule. | 7.4 |
| 5 — rejected | microsoft/trocr-base-str | 4,135 | MIT | transformers/optimum | Genuinely scene-text (STR), but a ViT-encoder + autoregressive GPT-style **seq2seq decoder**, hundreds of MB, generation loop — the exact LLM-scale, exotic-op, dynamic-shape profile that fights Melange. Rejected on size + Melange-fit. | 4.8 |

---

## Winners & why (defending the scene-text choice)

**Detector: PaddlePaddle/PP-OCRv5_mobile_det (DBNet, MobileNetV3).**
DBNet emits a per-pixel text-probability map, so downstream box extraction (DBPostProcess)
recovers *arbitrary-orientation quadrilaterals* — exactly what angled street signs need,
unlike an axis-aligned word detector. It is scene+doc trained, the newest PP-OCR mobile
detector, ~4.75 MB, and converts to a clean static opset-12 ONNX with only standard ops.

**Recognizer: PaddlePaddle/latin_PP-OCRv5_mobile_rec (SVTR-LCNet, CTC) — the scene-text pick.**
This is the crux requirement. Two things make it the right call:
1. **Scene-text fit (the hard requirement).** SVTR is a scene-text-recognition architecture
   benchmarked on IC13/IC15/SVT/CUTE; PP-OCR trains it on a large synthetic-scene + real
   corpus, so it reads angled/stylised sign fonts — not just clean documents. The **Latin**
   variant covers the Latin-script languages a traveler actually meets (FR/ES/DE/IT/PT/…),
   the strongest task fit for offline-roaming travel.
2. **Melange-fit via CTC (the export-safety requirement).** It is a **CTC** head, *not*
   autoregressive. The assignment warns PARSeq/ABINet-style recognizers carry
   autoregressive/exotic ops that break clean static-shape opset-12 export, and to fall back
   to a CTC scene recognizer if so. SVTR-LCNet CTC exports cleanly *by construction*:
   verified static [1,3,48,320]→[1,40,838], opset 12, all-standard ops (Conv/MatMul/Gemm/
   Softmax/Transpose — no Loop, no NonMaxSuppression, no autoregressive decode). We get
   scene-text quality **without** the PARSeq export trap, so no fallback was needed.

Trade-off accepted: a CTC recognizer is marginally weaker than PARSeq on the very hardest
curved/occluded text, and it decodes one crop at a time (Dart orchestrates the crop loop).
That is the deliberate price of a guaranteed-clean Melange artifact. Both models being PP-OCR
also means **one export recipe** for the whole OCR family (the batch rule).

License: both **Apache-2.0** — clean for ZETIC's GTM / trade-show distribution. (PP-OCR
weights are trained with Apache-2.0 PaddleOCR tooling — no AGPL entanglement, unlike the
Ultralytics YOLO family.)

---

## Export — DETECTOR (`ppocrv5_mobile_det.onnx`)
- Recipe: `export.py` (OCR/PaddleOCR family recipe — paddle2onnx opset 12 → onnxslim with
  fixed `--input-shapes x:1,3,736,736`). First OCR-family recipe; recorded for reuse.
- Artifact: `ppocrv5_mobile_det.onnx` (~4.75 MB).
- Input: float32 `x` **[1,3,736,736]**, NCHW. **BGR** channel order (PaddleOCR keeps cv2 BGR
  through NormalizeImage — do NOT swap to RGB). Per-channel normalization after /255:
  mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (ImageNet stats, applied in channel
  order 0,1,2). Live frame is letterboxed to the 736×736 square. (736 = PP-OCR TRT optimum,
  divisible by 32 for DBNet; tunable — larger improves small/distant sign text at latency cost.)
- Output: float32 `fetch_name_0` **[1,1,736,736]** — a single-channel **text-probability map
  in [0,1]** (final **Sigmoid is baked** into the graph). **DBPostProcess is NOT baked**:
  binarize (thresh 0.3), contour → box, box_thresh 0.6, unclip_ratio 1.5, then map boxes back
  through the letterbox inverse — all in Dart.
- Opset 12. **Static shapes confirmed** (`onnx.checker` passes; programmatic no-dynamic-axis
  assertion in export.py passes). Ops: Conv, ConvTranspose, Resize, GlobalAveragePool,
  HardSigmoid, Sigmoid, Add, Mul, Concat, Relu (all standard).

## Export — RECOGNIZER (`latin_ppocrv5_mobile_rec.onnx`)
- Recipe: same family recipe; onnxslim fixed `--input-shapes x:1,3,48,320` (**fixes the
  variable text width** → static shapes).
- Artifact: `latin_ppocrv5_mobile_rec.onnx` (~8.0 MB).
- Input: float32 `x` **[1,3,48,320]**, NCHW. **BGR** channel order. Normalization:
  (pixel/255 − 0.5)/0.5 → range **[−1, 1]**. Height fixed 48; each detected crop is
  aspect-preserving resized to height 48, width ≤ 320, then **right-padded with zeros to 320**.
- Output: float32 `fetch_name_0` **[1,40,838]** — 40 CTC time-steps × 838 classes,
  **Softmax baked** (probabilities). **CTC decode is NOT baked** (done in Dart, see below).
- **Charset / CTC classes (838):** index **0 = CTC blank** (skip); indices **1..836** =
  `latin_charset.txt` lines 1..836 (836 chars); index **837 = space `' '`**. Greedy CTC:
  argmax per step → collapse consecutive duplicates → drop blank → map indices to chars.
  Confidence = mean of the max prob over the kept (non-blank) steps.
  **The dictionary ships in this folder as `latin_charset.txt`** (836 lines, order preserved);
  Dart prepends blank@0 and appends space@837 to rebuild the full 838-class map.
- Opset 12. **Static shapes confirmed** (assertion in export.py passes). Ops: Conv, MatMul,
  Gemm, Softmax, Transpose, HardSigmoid, ReduceMean, Add/Sub/Mul/Div/Pow/Sqrt, Reshape,
  Slice, Squeeze/Unsqueeze, AveragePool, GlobalAveragePool — all standard; **no Loop / no
  autoregressive ops / no NonMaxSuppression**.

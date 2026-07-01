# SPEC: ReceiptScanner

> Stage-0 pre-draft. Everything the ONNX reveals is filled in. Only GATE-0 fields
> (Melange served name/version/shapes) are left as "[GATE 0 — human paste-back]".
> Product/display name is the worker's to finalize; suggested: **"Expensa"** (or
> "ReceiptIQ"). Folder + Melange model names stay `ReceiptScanner*` regardless.

## One-line pitch
Upload a photo of a receipt/invoice and extract line-items, total, merchant, and date
into structured fields — fully offline on-device — for field-sales / enterprise expense
workflows where connectivity is poor and receipt financials should not touch a third-party
cloud OCR.

## Model
Two Melange models (detect → recognize). See model_selection.md for the two-model
justification (single recognizer can't localize receipt lines; end-to-end OCRs are
autoregressive/dynamic → not Melange-fit).

### Model A — DETECTOR (text-line localization)
- Source (HF repo / origin): PaddlePaddle/PP-OCRv5_mobile_det (Apache-2.0)
  (exported ONNX `ppocrv5_mobile_det.onnx`, ~4.5 MB)
- Architecture: PP-OCRv5 mobile DB (Differentiable Binarization) detector, fully
  convolutional (Conv/BN/Resize/ConvTranspose/Sigmoid).
- Melange model name: [GATE 0 — human paste-back] (requested: ajayshah/ReceiptScannerDet)
- Melange version: [GATE 0 — human paste-back] (requested: 1)
- Input tensor: float32[1,3,960,640], NCHW. Preprocess: resize/letterbox to 960×640,
  `/255`, ImageNet-normalize mean [0.485,0.456,0.406] std [0.229,0.224,0.225],
  **channel order BGR** (PP-OCR DecodeImage img_mode=BGR), then CHW.
  - Served shape: [GATE 0 — human paste-back]
- Output tensor: float32[1,1,960,640] — DB text probability map, values in [0,1]
  (Sigmoid baked into graph). Each pixel = P(text) in the 960×640 letterboxed space.
  - Served shape: [GATE 0 — human paste-back]
- Post-processing baked into ONNX? No. DB box extraction (threshold → contours →
  min-area boxes → unclip) is pure Dart.
- Classes / labels: n/a (single-channel text/no-text map).
- modelMode to use and why: RUN_AUTO (default). No client mode steers a crashing
  artifact; GPU/MPSGraph traps are server-side ZETIC concerns. See CLAUDE.md §5.

### Model B — RECOGNIZER (line-image → text)
- Source (HF repo / origin): PaddlePaddle/en_PP-OCRv5_mobile_rec (Apache-2.0)
  (exported ONNX `en_ppocrv5_mobile_rec.onnx`, ~7.5 MB)
- Architecture: PP-OCRv5 mobile English SVTR-LCNet recognizer with a CTC head.
- Melange model name: [GATE 0 — human paste-back] (requested: ajayshah/ReceiptScannerRec)
- Melange version: [GATE 0 — human paste-back] (requested: 1)
- Input tensor: float32[1,3,48,320], NCHW. Preprocess: resize a single line crop to
  height 48 keeping aspect, right-pad to width 320 (pad with 0 after normalization),
  `/255` then `(x-0.5)/0.5` → range **[-1,1]**, **channel order BGR**, CHW.
  - Served shape: [GATE 0 — human paste-back]
- Output tensor: float32[1,40,438] — CTC posteriors (softmaxed), 40 time-steps × 438
  classes. Layout: [batch, T=40, C=438]; per time-step a probability distribution over
  the 438-class alphabet.
  - Served shape: [GATE 0 — human paste-back]
- Post-processing baked into ONNX? Softmax yes; greedy CTC decode No (pure Dart).
- Classes / labels: 438-class CTC alphabet. **blank = index 0**, dict chars = indices
  1..436, **space = index 437**. Decode table shipped as `rec_charset.json` (438-entry
  index-ordered) and `rec_dict.txt` (raw 436-char dict). Charset covers digits, A–Z/a–z,
  punctuation, currency symbols ($ ¢ ₤ ₹ ₽ …), %, checkmarks — receipt-appropriate.
- modelMode to use and why: RUN_AUTO (default), same reasoning as Model A.

## text+box OUTPUT CONTRACT (what the Dart layer consumes)
The two models together emit **boxes (from DET) + text per box (from REC)**. The Dart
structuring layer sits on top and is NOT a model:
1. DET map → list of oriented text-line boxes in image space (after letterbox inverse).
2. For each box: crop + deskew, feed REC, greedy-CTC-decode → `{text, confidence}`.
3. Result set = `List<RecognizedLine{ box: quad, text: String, conf: double }>`.
4. Structuring (regex + layout heuristics, worker-owned): cluster boxes into rows by
   y-overlap; sort L→R within a row; then
   - **line-items:** a row with a description on the left and a trailing price token
     `\d+[.,]\d{2}` on the right.
   - **total:** row matching `TOTAL|AMOUNT DUE|BALANCE|GRAND TOTAL` (case-insensitive),
     taking the largest / last currency value; guard against SUBTOTAL/TAX rows.
   - **merchant:** largest / top-most text block above the first line-item.
   - **date:** regex over common formats (`\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}`,
     `\d{4}-\d{2}-\d{2}`, `DD MON YYYY`).

## Input source
- **File upload** (photo of a receipt/invoice from gallery or camera-capture-to-file).
  Not a live camera stream — a single still image, so no per-frame orientation race.
- Decode uploaded image to RGB, then convert to the **BGR** channel order the models
  expect (see per-model preprocess). Handle EXIF orientation from the photo.
- No fixed sensor/sample-rate concerns (still-image pipeline).

## Pre-processing pipeline (ordered, exact)
### For DET (whole image, once)
1. Decode uploaded image (apply EXIF orientation) → RGB pixels.
2. Letterbox-resize the whole image to 960×640 (preserve aspect, pad); record scale +
   pad offsets for the inverse.
3. Convert to BGR channel order.
4. `/255`, then per-channel `(v-mean)/std` with mean [0.485,0.456,0.406],
   std [0.229,0.224,0.225] (applied in BGR channel positions as PP-OCR does).
5. Reorder to NCHW [1,3,960,640]; flatten to Float32List; `Tensor.float32List(..., shape:[1,3,960,640])`.

### For REC (per detected line box)
1. Crop the line quad from the ORIGINAL image; perspective-deskew to a horizontal strip.
2. Resize to height 48 keeping aspect; if resized width < 320 right-pad to 320, if > 320
   scale to fit 320 (PP-OCR RecResizeImg semantics).
3. Convert to BGR; `/255` then `(v-0.5)/0.5` → [-1,1].
4. NCHW [1,3,48,320]; Float32List; `Tensor.float32List(..., shape:[1,3,48,320])`.

## Post-processing pipeline (ordered, exact)
### DET → boxes
1. Read output map [1,1,960,640] (row-major HxW).
2. Binarize at prob threshold 0.3 (PP-OCR default `thresh`).
3. Connected-components / contour trace the binary map.
4. For each region: min-area rotated rectangle; drop if box score (mean prob inside) <
   0.6 (`box_thresh`); "unclip" (dilate) the box by ratio 1.5 (`unclip_ratio`).
5. Invert the letterbox (exact reverse of DET pre-processing) → boxes in image space.
### REC → text
6. Read output [1,40,438] as [T=40, C=438], **time-major**: step t is `t*438 .. t*438+437`.
7. Greedy CTC: per step argmax over 438 → index sequence; **collapse consecutive repeats**;
   **drop blank (index 0)**; map remaining indices via `rec_charset.json` (index 437 = space).
8. Confidence = mean of the max-probabilities over the non-blank kept steps.
### Structuring (see output contract above)
9. Row-cluster + regex/layout heuristics → {merchant, date, line-items[], total}.

## UI
- Left to the worker. Functional must-haves: show the uploaded receipt with detected line
  boxes overlaid; a structured results panel (merchant, date, itemized list, total); a
  per-field confidence indicator; total inference-latency readout (det + N×rec + Dart).

## Platform targets
- iOS 16.6+, Android minSdk 24.
- Known OS traps: FP32-GPU CoreML artifact can crash in MPSGraph on iOS/macOS 26.3+ — not
  client-fixable (no modelMode avoids it); read the SERVED target+apType from the native
  console and confirm it is not GPU on affected OS versions. Realistic non-crashing
  fallback is TFLITE_FP16/CPU (hundreds of ms), not NPU. This applies to BOTH models.
- Two-model note: REC runs once per detected line (N invocations per receipt) — batching
  is not available (static batch=1), so budget N sequential REC calls; watch cumulative
  latency and consider capping/queueing lines.

## Validation focus (Tier A traps most likely for THIS model)
- **Tensor layout (REC):** output [1,40,438] is **time-major** — step t occupies a
  contiguous 438-block. Decode against a hand-built matrix with one known argmax per step
  and assert the string; a transposed read silently produces plausible-but-wrong text.
- **CTC decode + charset:** blank=0, collapse-repeats-then-drop-blank order matters; space
  is index 437 (NOT in the raw dict). Test "aa[blank]b" → "ab", and an index that maps to
  space. Verify indices map through `rec_charset.json` exactly (off-by-one = whole-string shift).
- **Coordinate space / letterbox inverse (DET):** DB map is in 960×640 letterboxed space;
  round-trip a known box forward (letterbox) then inverse and assert it returns within
  tolerance. Boxes are pixel-space, not normalized.
- **Channel order (BGR):** both models expect **BGR** per PP-OCR's inference.yml
  (DecodeImage img_mode=BGR). Feeding RGB silently degrades accuracy. Confirm against a
  PaddleOCR reference decode of a known crop; assert the chosen channel order in a test.
- **Normalization:** DET uses ImageNet mean/std after /255; REC uses (x-0.5)/0.5 → [-1,1].
  Different per model — test each; wrong range silently degrades confidence.
- **Score/threshold semantics (DET):** DB uses prob-map threshold 0.3 and box_thresh 0.6
  and unclip 1.5; test the threshold boundary and that unclip expands the box.
- **REC resize/pad:** height-48 keep-aspect then pad-to-320; test that padding does not
  introduce spurious characters (padded region should decode to blank/space).
- **Orientation (upload):** apply EXIF orientation on decode; assert a known box maps
  correctly through the letterbox for the orientation actually used (still image, so no
  live-buffer rotation race — but EXIF is the analog trap here).

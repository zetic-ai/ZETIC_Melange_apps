# SPEC: LiveDocRedact  (Stage-0 Explorer stub — GATE-0 fields left blank)

> Pre-drafted by the Explorer. Everything the two ONNX files reveal is filled in.
> ONLY the GATE-0 paste-back fields (Melange name/version, served shapes) are blank,
> marked **[GATE 0 — human paste-back]**. The orchestrator finalizes this after upload.
> Suggested user-facing product name (worker sets at GATE 3, not GATE 0): **"RedactLens"**
> — folder/bundle-id/Melange names stay `LiveDocRedact` / `ajayshah/DocText*`.

## One-line pitch
Point the camera at an ID / passport / medical form; text fields are detected and read
**fully on-device**, and PII (name, DOB, ID number) is auto-boxed/blurred in the live
preview **before anything is stored or sent** — the whole pitch for fintech ID-scanner +
healthcare prospects who cannot stream documents to a cloud OCR.

## Pipeline architecture (binding — this is the integration design, not the worker's to invent)

```
 camera frame (BGRA iOS / YUV420 Android)
        │  letterbox → [1,3,640,640] BGR, ImageNet-norm
        ▼
 ┌────────────────────────────┐  ajayshah/DocTextDetector v1  (DBNet)
 │ 1. TEXT DETECTOR            │  x[1,3,640,640] → heatmap[1,1,640,640]
 └────────────────────────────┘
        │  DB decode (binarize thresh≈0.3 → contours → unclip≈1.5 → box_thresh≈0.6)
        │  → quad text-region boxes, un-letterboxed to screen space
        ▼
   text regions: [{quad, screen_bbox}, ...]     ("where is text")
        │  FOR EACH region: crop + rotate-upright + resize→[1,3,48,320] BGR, [-1,1]
        ▼
 ┌────────────────────────────┐  ajayshah/DocTextRecognizer v1  (CRNN/SVTR CTC)
 │ 2. TEXT RECOGNIZER (×N)     │  x[1,3,48,320] → logits[1,40,438]
 └────────────────────────────┘
        │  greedy CTC decode (argmax → collapse repeats → drop blank) via en_dict.txt
        ▼
   read fields: [{screen_bbox, text}, ...]
        │  PII classifier (pure-Dart regex/heuristics: name / DOB / ID-number / MRZ)
        ▼
   UI: blur/box the PII boxes live in the preview + "on-device · no cloud" badge
```

**The text-region grouping / crop-and-feed orchestration is Dart (worker-owned).** The two
Melange models are single forward passes; the loop that runs the recognizer once per detected
box, and any line-grouping, lives in the pipeline. PII detection is heuristic pure-Dart on the
recognized strings (regex for dates/ID patterns, keyword-anchored name fields, MRZ `<<` runs).

---

## Model

### Model 1 — Text detector
- Source (HF repo / origin): PaddlePaddle/PP-OCRv5_mobile_det_onnx (Apache-2.0), pre-exported
  ONNX, input pinned `[1,3,640,640]` + onnxslim-folded (see `export.py`).
- Architecture: **DBNet** (Differentiable-Binarization) mobile — fully-convolutional
  (Conv/BN/ConvTranspose/Sigmoid/Resize; source opset 11 upgraded to 12). ~4.75 MB.
- Melange model name: **[GATE 0 — human paste-back]** (requested: `ajayshah/DocTextDetector`)
- Melange version: **[GATE 0 — human paste-back]** (requested: 1)
- Input tensor: float32 **[1,3,640,640]**, NCHW, **BGR**, ImageNet-norm `(pixel/255 − mean)/std`,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
  - Served shape: **[GATE 0 — human paste-back]**
- Output tensor: float32 **[1,1,640,640]** — per-pixel text-probability heatmap (~0..1) in the
  640×640 letterboxed space. NOT a class tensor.
  - Served shape: **[GATE 0 — human paste-back]**
- Post-processing baked into ONNX? **No.** DB decode (binarize → contours → unclip → box filter)
  is pure-Dart.
- Classes / labels: N/A (heatmap).
- modelMode to use and why: **RUN_AUTO** (default). No client mode steers a crashing artifact;
  GPU/MPSGraph traps are handled server-side by ZETIC (CLAUDE.md §5).

### Model 2 — Text recognizer
- Source: PaddlePaddle/en_PP-OCRv5_mobile_rec (Apache-2.0), paddle2onnx (opset 12), input pinned
  to the **fixed** `[1,3,48,320]` + onnxslim-folded (see `export.py`).
- Architecture: **CRNN/SVTR CTC** English/Latin recognizer. ~7.8 MB.
- Melange model name: **[GATE 0 — human paste-back]** (requested: `ajayshah/DocTextRecognizer`)
- Melange version: **[GATE 0 — human paste-back]** (requested: 1)
- Input tensor: float32 **[1,3,48,320]**, NCHW, **BGR**, PP-OCR rec norm `(pixel/255 − 0.5)/0.5`
  → **[-1,1]**. **Fixed width 320** (variable width is a static-shape violation).
  - Served shape: **[GATE 0 — human paste-back]**
- Output tensor: float32 **[1,40,438]** — 40 CTC time-steps × 438 classes (softmax probabilities).
  - Served shape: **[GATE 0 — human paste-back]**
- Post-processing baked into ONNX? **No.** Greedy CTC decode is pure-Dart.
- Classes / labels: **CTC label list = [blank](0) + 436 chars from `en_dict.txt` (idx 1..436) +
  space ' '(idx 437)**. `en_dict.txt` ships in this folder — the Dart decoder MUST load it and
  build the list in exactly this order.
- modelMode to use and why: **RUN_AUTO** (default).

---

## Input source
- Rear camera, cheapest usable pixel format (BGRA on iOS, YUV420 on Android → convert to BGR).
- Both models expect **BGR** channel order (PaddleOCR cv2 convention) — see orientation/channel trap.
- Orientation handling required: measure the real buffer WxH on-device; on the PyroGuard iOS setup
  the BGRA buffer arrived **upright (720×1280)** needing **NO** rotation. Do NOT assume landscape —
  the real PyroGuard bug was a *spurious* 90° rotation. Text OCR is orientation-sensitive twice over
  (see traps).

## Pre-processing pipeline (ordered, exact)
**Detector (once per frame):**
1. Capture frame bytes → convert source pixel format to **BGR** (drop alpha; YUV→BGR on Android).
2. Letterbox-resize to 640×640 preserving aspect (record scale + pad for the inverse).
3. Normalize per channel `(pixel/255 − mean)/std`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`
   — applied index-wise to the **BGR** channels exactly as PaddleOCR does (do not reorder to RGB).
4. Reorder to NCHW [1,3,640,640]; flatten to Float32List; `Tensor.float32List(data, shape:[1,3,640,640])`.

**Recognizer (once PER detected region):**
5. Crop the region quad from the ORIGINAL frame (not the letterboxed tensor) and **deskew/rotate the
   quad upright** (perspective/affine warp so text reads left-to-right, height→48).
6. Resize keeping aspect to H=48; if resulting W < 320 **right-pad with zeros to 320**, if W > 320
   downscale to 320. (Fixed-width static resolution.)
7. Normalize `(pixel/255 − 0.5)/0.5` → [-1,1] on **BGR**; NCHW [1,3,48,320]; wrap as Tensor.

## Post-processing pipeline (ordered, exact)
**Detector → text regions (DBPostProcess: `thresh≈0.3`, `box_thresh≈0.6`, `unclip_ratio≈1.5`,
`max_candidates≈1000`):**
1. Read heatmap [1,1,640,640] (values ~0..1; a Sigmoid is baked into the graph — apply NO extra sigmoid).
2. Binarize at `thresh`; find connected components / contours (min-area-rect per blob).
3. Filter each box by mean heatmap score ≥ `box_thresh`; **unclip** (dilate) the polygon by `unclip_ratio`.
4. **Undo the letterbox** (exact reverse of pre-step 2) to map each quad back to screen/original space.
5. Emit `TextRegion{quad, screen_bbox}`. (Optional: group boxes into reading lines — Dart.)

**Recognizer → text (greedy CTC, per region):**
6. Read logits [1,40,438]; for each of the 40 steps take `argmax` over the 438 classes.
7. **Collapse consecutive duplicates**, then **drop blank (class 0)**; map remaining indices via the
   label list `[blank]+en_dict.txt+[' ']` → characters → the region's string.
8. Emit `ReadField{screen_bbox, text}`.

**PII redaction (pure-Dart heuristics):**
9. Classify each `ReadField.text` as name / DOB / ID-number / MRZ / other (regex + keyword anchors).
10. Blur/box the screen_bbox of PII fields in the live preview; nothing is persisted or transmitted.

## UI
- Left to the worker. Functional must-haves: live overlay of detected text boxes; **PII boxes
  blurred/redacted in the preview**; a visible **"on-device · no cloud"** badge (airplane-mode-friendly);
  inference-latency / regions-per-frame HUD (shown on-screen, since Dart `print` won't reach the release
  device console — CLAUDE.md §5).

## Platform targets
- iOS 16.6+, Android minSdk 24 (PyroGuard baseline).
- Known OS traps: (a) iOS/macOS 26.3+ CoreML-GPU MPSGraph crash — handled server-side by ZETIC; confirm
  the *served* artifact isn't GPU on affected OS via the device console. (b) "Benchmarked ≠ served" —
  budget CPU-speed until `runtimeApType=NPU` is confirmed. (c) **Two models + N recognizer calls per
  frame** → cold-start is 2 model loads and the hot path is 1 det + N rec inferences; warm BOTH with a
  dummy inference right after load, and throttle/`_busy`-guard frames.

## Validation focus (Tier-A traps most likely for THESE models — worker must cover with tests)
- **CTC decode semantics (recognizer, #1 silent-wrong):** blank is index **0**; label list is exactly
  `[blank] + en_dict.txt(436) + [' ']` (space is the LAST class, 437). Test: hand-built [1,40,438] logits
  encoding a known string with repeats + blanks → assert collapse-repeats-then-drop-blank yields the string.
- **Recognizer output layout:** [1,40,438] is (steps, classes) — argmax is over the 438 (last) axis per
  step, NOT across steps. Test against a hand-built tensor with one hot class per step.
- **Fixed-width padding round-trip (recognizer):** a crop narrower than 320 must be right-padded with
  zeros (not stretched); assert aspect-preserving resize + pad reproduces H=48,W=320 and that padding
  doesn't emit spurious characters.
- **Detector heatmap decode + letterbox inverse:** round-trip a known box through forward letterbox →
  DB unclip → inverse letterbox and assert it returns to the original within tolerance. Coordinates are
  in **640×640 pixel space**, not normalized.
- **Score/activation semantics:** the detector heatmap is already Sigmoid-activated in-graph and the
  recognizer head is already Softmax'd — apply NO extra activation in Dart. Test against the real tensors.
- **Channel order (BGR vs RGB):** both models were trained on cv2 **BGR** with the mean/std applied
  index-wise to BGR; a silent R/B swap degrades accuracy without throwing. Assert the preprocessor keeps
  BGR order and verify recognized text on a known on-device sample.
- **Orientation (twice):** (i) the full-frame buffer orientation into the detector — measure real buffer
  WxH; do not assume landscape (PyroGuard's bug was a *spurious* rotation). (ii) per-crop deskew — a text
  quad fed sideways/upside-down to the recognizer returns garbage; assert the quad-warp makes text upright.
- **Text-region grouping / geometry:** two adjacent fields must not merge into one box (unclip too large)
  nor split a word; test DB box extraction + optional line-grouping on a hand-built heatmap.
- **Threshold boundaries:** detector `box_thresh` (just-below dropped, just-above kept); recognizer
  low-confidence step handling.

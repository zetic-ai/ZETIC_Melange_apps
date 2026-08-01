# SPEC: LiveDocRedact  (FINAL — GATE 0 passed, registered models injected)

> Finalized from `SPEC_stub.md` + the human's GATE-0 dashboard paste-back (Jul 2 2026).
> **GATE-0 rename:** the models registered as `ajayshah/LiveDocRedact_Detect` and
> `ajayshah/LiveDocRedact_Recognize` — NOT the Explorer's proposed `ajayshah/DocTextDetector`
> / `ajayshah/DocTextRecognizer`. This spec uses the registered names everywhere; ignore the
> old names in `SPEC_stub.md` / `melange_upload.md` / `model_selection.md`.
> Suggested user-facing product name (worker sets at GATE 3): **"RedactLens"** — folder,
> bundle id, and the registered Melange model names stay as above.

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
 ┌────────────────────────────┐  ajayshah/LiveDocRedact_Detect v1  (DBNet)
 │ 1. TEXT DETECTOR            │  x[1,3,640,640] → heatmap fetch_name_0[1,1,640,640]
 └────────────────────────────┘
        │  DB decode (binarize thresh≈0.3 → contours → unclip≈1.5 → box_thresh≈0.6)
        │  → quad text-region boxes, un-letterboxed to screen space
        ▼
   text regions: [{quad, screen_bbox}, ...]     ("where is text")
        │  FOR EACH budgeted region: crop + rotate-upright + resize→[1,3,48,320] BGR, [-1,1]
        ▼
 ┌────────────────────────────┐  ajayshah/LiveDocRedact_Recognize v1  (CRNN/SVTR CTC)
 │ 2. TEXT RECOGNIZER (×N)     │  x[1,3,48,320] → logits fetch_name_0[1,40,438]
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

### Frame-flow latency budget (binding — plan for the CPU fallback, per CLAUDE.md §5)
Dashboard benchmarks (GATE-0 paste-back):

| Model | NPU (min/med/avg) | GPU med | CPU med | Deployability | Size (served) |
|---|---|---|---|---|---|
| LiveDocRedact_Detect | 2.78 / 5.79 / 9.14 ms | 164 ms | 128.6 ms | 98% (FP32 100%) | 1.31–4.99 MB, 3 quants |
| LiveDocRedact_Recognize | 0.52 / 1.26 / — ms | 48.9 ms | 31.9 ms | 98% (FP32 100%) | 2.03–8.03 MB, 3 quants |

- **NPU numbers are the ideal, not the plan.** "Benchmarked ≠ served" (CLAUDE.md §5): until
  `runtimeApType=NPU` is confirmed on the device console, budget the realistic CPU fallback:
  **~129 ms detector + ~32 ms recognizer PER CROP, × N crops per frame.** A document frame can
  yield tens of text regions → a naive run-everything frame is seconds on CPU.
- **Therefore the recognizer runs on a BUDGET (binding).** The worker must implement one (or
  both) of:
  - **Top-K per frame:** recognize only the K highest-score / largest detector regions per
    frame (K tunable, e.g. 3–5 on CPU, raise when NPU confirmed), prioritizing not-yet-read
    regions; or
  - **Staggered recognition:** run detection every frame (or every Nth frame) for live boxes,
    and spread recognizer calls across subsequent frames round-robin, caching recognized text
    per region (keyed by IoU-matched box) so already-read fields keep their PII state without
    re-running.
  Detector boxes (and cached PII blurs) stay live-overlaid every processed frame either way,
  so the demo remains responsive even when text for a region arrives a few frames late.
- Standard frame guards: `_busy` flag + drop (never queue) frames; warm BOTH models with one
  dummy inference right after load; show per-stage timings on the HUD.

---

## Model

### Model 1 — Text detector
- Source (HF repo / origin): PaddlePaddle/PP-OCRv5_mobile_det_onnx (Apache-2.0), pre-exported
  ONNX, input pinned `[1,3,640,640]` + onnxslim-folded (see `export.py`).
- Architecture: **DBNet** (Differentiable-Binarization) mobile — fully-convolutional
  (Conv/BN/ConvTranspose/Sigmoid/Resize; source opset 11 upgraded to 12). ~4.75 MB ONNX;
  served 1.31–4.99 MB across 3 quantizations.
- Melange model name: **`ajayshah/LiveDocRedact_Detect`** — CONFIRMED at GATE 0. The dashboard
  displays it as "ZETIC | LiveDocRedact_Detect"; `ZETIC |` is the org display prefix, NOT the
  account — the SDK `create()` name is exactly `ajayshah/LiveDocRedact_Detect` (with the slash;
  a bare name throws `MlangeException(3)` on-device, per CLAUDE.md §5).
- Melange version: **v1** (assumed — first upload, dated Jul 2 2026; **confirm at first SDK
  create**).
- Input tensor: float32 **[1,3,640,640]**, NCHW, **BGR**, ImageNet-norm `(pixel/255 − mean)/std`,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
  - Served input: **`x` float32[1,3,640,640]** — CONFIRMED, matches export exactly.
- Output tensor: float32 **[1,1,640,640]** — per-pixel text-probability heatmap (~0..1) in the
  640×640 letterboxed space. NOT a class tensor.
  - Served output: **`fetch_name_0` float32[1,1,640,640]** — CONFIRMED, matches export exactly.
- Benchmark (dashboard): NPU min 2.78 / med 5.79 / avg 9.14 ms; GPU med 164 ms; CPU med
  128.6 ms. Deployability 98%, FP32 100%.
  - ⚠️ **Dashboard accuracy row for `fetch_name_0` shows 0.00 dB min/max.** Likely a report
    artifact (heatmap output confuses the SNR metric), but do NOT assume the benchmarked row
    is what gets served: the on-device sanity check in Validation focus / Tier C must verify a
    real, non-degenerate heatmap (varying values, text regions light up) before trusting it.
- Post-processing baked into ONNX? **No.** DB decode (binarize → contours → unclip → box filter)
  is pure-Dart.
- Classes / labels: N/A (heatmap).
- modelMode to use and why: **RUN_AUTO**. Per CLAUDE.md §5, no client mode steers off a bad
  artifact (all four modes returned the same artifact on PyroGuard); backend selection is
  server-side, and the *served* artifact read from the native console is ground truth.

### Model 2 — Text recognizer
- Source: PaddlePaddle/en_PP-OCRv5_mobile_rec (Apache-2.0), paddle2onnx (opset 12), input pinned
  to the **fixed** `[1,3,48,320]` + onnxslim-folded (see `export.py`).
- Architecture: **CRNN/SVTR CTC** English/Latin recognizer. ~7.8 MB ONNX; served 2.03–8.03 MB
  across 3 quantizations.
- Melange model name: **`ajayshah/LiveDocRedact_Recognize`** — CONFIRMED at GATE 0 (dashboard
  "ZETIC | LiveDocRedact_Recognize"; SDK name is the `ajayshah/...` form with the slash).
- Melange version: **v1** (assumed — first upload; **confirm at first SDK create**).
- Input tensor: float32 **[1,3,48,320]**, NCHW, **BGR**, PP-OCR rec norm `(pixel/255 − 0.5)/0.5`
  → **[-1,1]**. **Fixed width 320** (variable width is a static-shape violation).
  - Served input: **`x` float32[1,3,48,320]** — CONFIRMED, matches export exactly.
- Output tensor: float32 **[1,40,438]** — 40 CTC time-steps × 438 classes (softmax probabilities).
  - Served output: **`fetch_name_0` float32[1,40,438]** — CONFIRMED, matches export exactly.
- Benchmark (dashboard): NPU min 0.52 / med 1.26 ms; GPU med 48.9 ms; CPU med 31.9 ms;
  accuracy 4.76–29.21 dB across quantizations; deployability 98%, FP32 100%.
- Post-processing baked into ONNX? **No.** Greedy CTC decode is pure-Dart.
- Classes / labels: **CTC label list = [blank](0) + 436 chars from `en_dict.txt` (idx 1..436) +
  space ' '(idx 437)**. `en_dict.txt` ships in this folder — the Dart decoder MUST load it and
  build the list in exactly this order.
- modelMode to use and why: **RUN_AUTO** (same rationale as the detector).

---

## Input source
- Rear camera, cheapest usable pixel format (BGRA on iOS, YUV420 on Android → convert to BGR).
- Both models expect **BGR** channel order (PaddleOCR cv2 convention) — see orientation/channel trap.
- Orientation handling required: measure the real buffer WxH on-device; on the PyroGuard iOS setup
  the BGRA buffer arrived **upright (720×1280)** needing **NO** rotation. Do NOT assume landscape —
  the real PyroGuard bug was a *spurious* 90° rotation. Text OCR is orientation-sensitive twice over
  (see traps).

## Pre-processing pipeline (ordered, exact)
**Detector (once per processed frame):**
1. Capture frame bytes → convert source pixel format to **BGR** (drop alpha; YUV→BGR on Android).
2. Letterbox-resize to 640×640 preserving aspect (record scale + pad for the inverse).
3. Normalize per channel `(pixel/255 − mean)/std`, mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`
   — applied index-wise to the **BGR** channels exactly as PaddleOCR does (do not reorder to RGB).
4. Reorder to NCHW [1,3,640,640]; flatten to Float32List; `Tensor.float32List(data, shape:[1,3,640,640])`.

**Recognizer (once PER budgeted region — see frame-flow latency budget):**
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
10. Blur/box the screen_bbox of PII fields in the live preview (including cached fields whose text
    arrived on an earlier frame, under the staggered/top-K budget); nothing is persisted or transmitted.

## UI
- Left to the worker (visual design is the worker's choice). Functional must-haves:
  - Live overlay of detected text boxes; **PII boxes blurred/redacted in the preview**.
  - A visible **"on-device · no cloud"** badge (airplane-mode-friendly).
  - Inference-latency / regions-per-frame HUD, including per-stage timings (det ms, rec ms/crop,
    crops recognized this frame) — shown on-screen, since Dart `print` won't reach the release
    device console (CLAUDE.md §5).
  - Per-class live counts of PII fields found (name / DOB / ID-number / MRZ).

## Platform targets
- iOS 16.6+, Android minSdk 24 (PyroGuard baseline).
- Known OS traps:
  - (a) iOS/macOS 26.3+ CoreML-GPU MPSGraph crash history — an FP32-GPU artifact can load
    cleanly then SIGABRT at first inference; not client-fixable (no modelMode avoids it) and
    handled server-side by ZETIC filtering GPU. Confirm the *served* artifact isn't GPU on
    affected OS versions via the native device console.
  - (b) "Benchmarked ≠ served" — budget CPU-speed (det ~129 ms, rec ~32 ms/crop) until
    `runtimeApType=NPU` is confirmed on the device console.
  - (c) Use **release builds on device** (debug hangs on recent iOS/Xcode); observability lives
    in the native console (`xcrun devicectl device process launch --console ...`), and Dart
    `print` does not reach it in release — put diagnostics on the HUD.
  - (d) **Two models + N recognizer calls per frame** → cold start is 2 model loads + 2 model
    downloads on first launch; warm BOTH with a dummy inference right after load, and
    throttle/`_busy`-guard frames.

## Validation focus (per VALIDATION.md — Tier A tests the worker must write; Tier B/C obligations noted)

**Tier A (autonomous unit tests, hand-built data, must pass before GATE 3):**
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
- **Recognizer-budget scheduling:** test the top-K / staggered scheduler with a mock region list —
  assert K-cap is respected per frame, unread regions are prioritized, and cached text is reused
  (IoU-matched) rather than re-recognized.

**Tier B (0.5% rule):** the A4 hot-path micro-benchmark must cover the full per-frame pipeline
at realistic N (detector decode + K recognizer preprocess/decode passes); every optimization
shows a ≥0.5% before/after delta on it (see VALIDATION.md Tier B).

**Tier C (surface to the human at GATE 3, per VALIDATION.md):**
- **Detector accuracy-0dB on-device sanity check (from GATE 0):** the dashboard accuracy row
  for `fetch_name_0` reads 0.00 dB min/max — likely a report artifact, but the served artifact
  is not necessarily the benchmarked row. Before trusting it, verify on-device that the served
  detector returns a real, NON-DEGENERATE heatmap: HUD-display heatmap min/max/mean and confirm
  values vary and text regions light up (not all-zeros / all-constant) on a known document.
- Served artifact per model: read `runtimeApType` (+ target/precision) from the native console
  for BOTH models; NPU is the ideal, CPU the realistic fallback — the recognizer budget (top-K
  K value / stagger rate) should be re-tuned to whichever backend is actually served.
- Standard Tier-C items: modelMode recorded as RUN_AUTO (not a crash workaround); exact
  device-console command wired before the first run; signing/Developer-Mode gates; release
  build; first-launch double model download on conference Wi-Fi (pre-warm); multi-cold-start
  acceptance (server-side selection is non-deterministic); personal key embedded in client.

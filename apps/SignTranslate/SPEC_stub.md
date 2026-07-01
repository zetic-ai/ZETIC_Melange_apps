# SPEC: SignTranslate

## One-line pitch
LIVE camera, fully-offline scene-text reader for a traveler with no signal / no roaming
data: point the phone at a street sign, menu, or label and read the text live as the camera
moves — with an OPTIONAL on-device translate step. Two on-device models (text detector +
scene-text recognizer) through the ZETIC Melange SDK.

## Pipeline shape
TWO Melange models, both required, run per frame:
1. **Detector** (`ajayshah/SceneTextDetector`) → text-probability map → Dart extracts text-region quads.
2. Dart crops + normalizes each region.
3. **Recognizer** (`ajayshah/SceneTextRecognizer`) → per-crop CTC logits → Dart greedy-decodes to a string.
4. (OPTIONAL, DOWNSTREAM, out of scope for Stage 0) local translate of the decoded string — Dart, no ML model exported here.

---

## Model A — text DETECTOR
- Source (HF repo / origin): PaddlePaddle/PP-OCRv5_mobile_det (Apache-2.0)
  (exported ONNX `ppocrv5_mobile_det.onnx`, ~4.75 MB)
- Architecture: DBNet + MobileNetV3 (differentiable-binarization text detector), scene+doc trained
- Melange model name: `[GATE 0 — human paste-back]` (requested: ajayshah/SceneTextDetector)
- Melange version: `[GATE 0 — human paste-back]` (requested: 1)
- Input tensor: float32[1,3,736,736], NCHW, **BGR** channel order; per-channel normalize after
  /255: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (ImageNet stats)
  - Served shape: `[GATE 0 — human paste-back]`
- Output tensor: float32[1,1,736,736] — single-channel text-probability map, values [0,1]
  (final **Sigmoid baked** into the graph)
  - Served shape: `[GATE 0 — human paste-back]`
- Post-processing baked into ONNX? Sigmoid: YES (baked). Box extraction (DBPostProcess):
  NO — binarize/contour/unclip in pure Dart.
- Classes / labels: n/a (dense probability map)
- modelMode to use and why: RUN_AUTO (default). No client mode steers a crashing artifact;
  GPU/MPSGraph traps are handled server-side by ZETIC. See CLAUDE.md S5.

## Model B — text RECOGNIZER (scene-text)
- Source (HF repo / origin): PaddlePaddle/latin_PP-OCRv5_mobile_rec (Apache-2.0)
  (exported ONNX `latin_ppocrv5_mobile_rec.onnx`, ~8.0 MB)
- Architecture: SVTR-LCNet, **CTC** head (scene-text recognizer; NOT autoregressive)
- Melange model name: `[GATE 0 — human paste-back]` (requested: ajayshah/SceneTextRecognizer)
- Melange version: `[GATE 0 — human paste-back]` (requested: 1)
- Input tensor: float32[1,3,48,320], NCHW, **BGR** channel order; normalize (pixel/255 − 0.5)/0.5
  → range [−1,1]. Height fixed 48; crop aspect-preserving resized to height 48, width ≤ 320,
  then **right-padded with zeros to 320**.
  - Served shape: `[GATE 0 — human paste-back]`
- Output tensor: float32[1,40,838] — 40 CTC time-steps × 838 classes, **Softmax baked** (probs)
  - Served shape: `[GATE 0 — human paste-back]`
- Post-processing baked into ONNX? Softmax: YES (baked). CTC greedy decode: NO — in Dart.
- Classes / labels: 838 CTC classes — index **0 = blank** (skip); indices **1..836** =
  `latin_charset.txt` lines 1..836; index **837 = space `' '`**. Dictionary shipped as
  `latin_charset.txt` (836 lines, order preserved) in this folder — the Dart asset.
- modelMode to use and why: RUN_AUTO (default). Same rationale as Model A.

---

## Input source
- Rear camera, LIVE, cheapest usable pixel format (BGRA on iOS, YUV420 on Android → convert).
  Note: both models expect **BGR** channel order (PaddleOCR convention) — do NOT swap to RGB.
- Orientation handling required: measure the real buffer WxH on-device (HUD it). On the
  PyroGuard iOS setup the BGRA buffer arrived UPRIGHT (720x1280) needing NO rotation — do not
  assume landscape. Verify, then map detected quads back to screen space accordingly.
  Scene text is arbitrary-orientation, so an upright buffer is fine — DBNet finds the angled
  regions; the recognizer receives axis-aligned crops after Dart de-skews each quad.

## Pre-processing pipeline (ordered, exact)
**Detector (once per frame):**
1. Capture frame bytes; convert source pixel format → BGR (drop alpha; keep BGR order).
2. Letterbox-resize the frame to 736×736 preserving aspect (pad; remember scale + pad offsets
   for the inverse). 736 is divisible by 32 (DBNet requirement).
3. Normalize per channel after /255: subtract [0.485,0.456,0.406], divide [0.229,0.224,0.225].
4. Reorder to NCHW [1,3,736,736]; flatten to Float32List; wrap Tensor.float32List(shape:[1,3,736,736]).

**Recognizer (once per detected text region):**
5. Warp/crop each detected quad to an upright rectangle (de-skew the angled region).
6. Aspect-preserving resize to height 48, width = min(round(48·w/h), 320).
7. Right-pad with zeros to width 320 → [3,48,320], BGR.
8. Normalize (pixel/255 − 0.5)/0.5 → [−1,1]; NCHW [1,3,48,320]; wrap as Tensor.float32List.

## Post-processing pipeline (ordered, exact)
**Detector output [1,1,736,736] (probability map, [0,1], Sigmoid already applied — apply NO
extra sigmoid):**
1. Binarize at prob threshold 0.3.
2. Find connected components / contours of the binary map.
3. For each candidate: compute the mean probability inside; drop if < box_thresh 0.6.
4. Fit a minimum-area rotated box; **unclip** (dilate) by unclip_ratio 1.5 to recover full glyphs.
5. Undo the letterbox (exact reverse of pre-proc steps 2) to map each quad into screen space.
6. Emit a list of text-region quads (ordered top→bottom, left→right for reading order).

**Recognizer output [1,40,838] per crop (Softmax already applied — apply NO extra softmax):**
7. For each of 40 time-steps: argmax over the 838 classes → index sequence (length 40).
8. Collapse consecutive duplicate indices (CTC merge).
9. Drop blank (index 0).
10. Map each remaining index → char: 1..836 → `latin_charset.txt`[idx−1]; 837 → space.
11. Confidence = mean of the per-step max prob over the kept (non-blank) steps.
12. Emit RecognizedText{ quad, string, conf }. (Optional downstream: local translate of `string`.)

## UI
- Left to the worker. Functional must-haves: live overlay of detected text quads with the
  decoded string + confidence anchored to each region; live count of regions read;
  inference-latency readout (detector ms + recognizer ms, or total). An OPTIONAL toggle to
  translate the recognized strings locally (downstream, not part of the ML models).

## Platform targets
- iOS 16.6+, Android minSdk 24.
- Known OS traps: FP32-GPU CoreML artifact can crash in MPSGraph on iOS/macOS 26.3+; not
  client-fixable (no modelMode avoids it) — read the SERVED target+apType from the native
  console and confirm it is not GPU on affected OS versions. Realistic non-crashing fallback is
  TFLITE_FP16/CPU (hundreds of ms), not the NPU. This matters more here: TWO model runs per
  frame (detector + N recognizer crops), so budget latency accordingly and consider throttling
  recognizer calls (e.g. only re-read regions that changed).

## Validation focus (Tier A traps most likely for THESE models)
- **Tensor layout — recognizer [1,40,838]:** decode against a hand-built logit tensor where
  one known time-step sequence spells a known short word; assert time-major stepping over 40
  and class over 838, and that the CTC merge+blank-strip yields the exact string.
- **CTC charset indexing (the #1 silent-wrong trap):** assert blank=index 0 is skipped,
  index i (1..836) maps to `latin_charset.txt`[i−1], and index 837 → space. An off-by-one
  here shifts every character.
- **Score semantics:** confirm Softmax is ALREADY baked in the recognizer and Sigmoid ALREADY
  baked in the detector — apply NO extra activation in Dart. Test against the real tensors.
- **Detector threshold behavior:** just-below prob-threshold pixel dropped, just-above kept;
  and box_thresh mean-probability boundary for a region.
- **Letterbox / resize inverse (detector):** round-trip a known quad — forward letterbox to
  736×736, inverse back — assert it returns within tolerance. Inverse must be the exact
  reverse order of the forward steps.
- **Coordinate space:** detector map is in 736×736 letterboxed pixel space; recognizer crop is
  in its own 48×320 padded space — the right-pad must not be read as characters (blank region
  → blank steps).
- **Channel order:** both models expect **BGR** (PaddleOCR convention), not RGB — a swap
  silently degrades accuracy. Assert the pre-proc feeds BGR.
- **Orientation:** assert the chosen transform round-trips a known quad for the buffer
  orientation actually measured on-device (HUD the buffer WxH + one raw quad).
- **Crop-feed correctness:** a de-skewed angled crop fed to the recognizer must recover the
  same string as its axis-aligned version (scene-text robustness spot-check).

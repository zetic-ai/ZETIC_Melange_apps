# SPEC: SignTranslate — FINAL (GATE-0 confirmed)

> Finalized 2026-07-02 with GATE-0 dashboard paste-back. This spec supersedes
> `SPEC_stub.md`. The two Melange models are REGISTERED as
> `ajayshah/SignTranslate_Detect` and `ajayshah/SignTranslate_Rec` — these names
> replace the Explorer's proposed `ajayshah/SceneTextDetector` /
> `ajayshah/SceneTextRecognizer` everywhere (stub and `melange_upload.md` are
> historical; this file is authoritative).

## One-line pitch
LIVE camera, fully-offline scene-text reader for a traveler with no signal / no roaming
data: point the phone at a street sign, menu, or label and the text is read live, tracked
to the regions as the camera moves. Two on-device models (text detector + scene-text
recognizer) through the ZETIC Melange SDK. An optional local translate step is
DOWNSTREAM and out of scope for the ML pipeline (no model; a dictionary/phrase-table
option is out of scope for the v1 UI unless trivial).

**Product display name (worker applies as display name ONLY):** **GlyphGo**.
Set iOS `CFBundleDisplayName`, Android `android:label`, and the in-app title
(`MaterialApp(title:)`, app bar / loading text). Do NOT change the bundle id, the
`apps/SignTranslate/` folder name, or the registered Melange model names.

## Pipeline shape
TWO Melange models, both required:
1. **Detector** (`ajayshah/SignTranslate_Detect`) → text-probability heatmap → Dart
   DBPostProcess extracts text-region quads.
2. Dart deskews + crops + normalizes each region (quad warp upright).
3. **Recognizer** (`ajayshah/SignTranslate_Rec`) → per-crop CTC probs → Dart greedy CTC
   decode → string.
4. (OPTIONAL, DOWNSTREAM) local translate of the decoded string — pure Dart, no ML model.

---

## Model A — text DETECTOR

- Source (HF repo / origin): `PaddlePaddle/PP-OCRv5_mobile_det` (Apache-2.0), exported
  ONNX `ppocrv5_mobile_det.onnx` (~4.75 MB), opset 12, static shapes (see `export.py`,
  `model_selection.md`).
- Architecture: DBNet + MobileNetV3 (differentiable-binarization text detector),
  scene + document trained, arbitrary-orientation text regions.
- Melange model name: **`ajayshah/SignTranslate_Detect`** — CONFIRMED registered at
  GATE 0. The dashboard header shows "ZETIC | SignTranslate_Detect"; that `ZETIC |` is a
  workspace DISPLAY prefix, NOT the account. The SDK `create(name:)` string is exactly
  `ajayshah/SignTranslate_Detect` WITH the slash (a bare name throws `MlangeException(3)`
  on-device at load; see CLAUDE.md §5).
- Melange version: **1** (assumed — first upload Jul 2 2026; **confirm at first SDK
  create**). This is the only value in this spec carrying an "assumed" caveat.
- Input tensor (served, CONFIRMED — matches export exactly): `x`
  float32[1,3,736,736], NCHW, **BGR** channel order (PaddleOCR keeps cv2 BGR — do NOT
  swap to RGB); values: /255 then per-channel ImageNet normalize
  mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] (applied in channel order 0,1,2).
- Output tensor (served, CONFIRMED — matches export exactly): `fetch_name_0`
  float32[1,1,736,736] — single-channel text-probability map in [0,1], in the 736×736
  letterboxed pixel space. Dim layout: [batch=1, channel=1, H=736, W=736].
- Post-processing baked into ONNX? Final **Sigmoid: YES (baked)** — apply NO extra
  sigmoid. **DBPostProcess (binarize/contour/unclip/box fit): NO** — pure Dart.
- Classes / labels: n/a (dense probability map, not a classifier).
- modelMode to use and why: **RUN_AUTO**. Per CLAUDE.md §5, no client mode steers which
  artifact is served (all four modes returned the same artifact on PyroGuard);
  backend/precision selection is server-side. The served artifact read from the native
  console is ground truth — record the requested mode but trust the console.
- GATE-0 benchmark (dashboard, informational — "benchmarked ≠ served"):
  NPU min 3.80 / med 7.35 / avg 12.86 ms; GPU med 219 ms; CPU med 169.1 ms.
  Deployability 98%, FP32 100%. Served size 1.31–4.99 MB across 3 quantizations.
  Load memory up to ~280 MB, inference up to ~352 MB.
- ⚠️ **Accuracy-row anomaly (carrying into Validation focus + Tier C):** the dashboard
  accuracy row for `fetch_name_0` reads **−4.12 to 0.00 dB** — anomalous. Most likely the
  SNR metric degenerating on a probability-heatmap output (the LiveDocRedact detector
  showed the same 0-dB artifact), NOT proof of a broken model — but it must be verified
  on-device: before trusting the detector, confirm the served model returns a real,
  NON-DEGENERATE heatmap (HUD the map's min/max/mean; text regions must visibly light
  up on a known sign). Benchmarked ≠ served.

## Model B — text RECOGNIZER (scene-text)

- Source (HF repo / origin): `PaddlePaddle/latin_PP-OCRv5_mobile_rec` (Apache-2.0),
  exported ONNX `latin_ppocrv5_mobile_rec.onnx` (~8.0 MB), opset 12, static shapes.
- Architecture: SVTR-LCNet with **CTC** head (scene-text recognizer — NOT
  autoregressive). Latin charset → FR/ES/DE/IT/PT and other Latin-script traveler
  languages, including accented characters.
- Melange model name: **`ajayshah/SignTranslate_Rec`** — CONFIRMED registered at GATE 0.
  Note it is **`_Rec`, NOT `_Recognize`** (and not the proposed `SceneTextRecognizer`).
  Dashboard shows "ZETIC | SignTranslate_Rec"; SDK name is
  `ajayshah/SignTranslate_Rec` with the slash. Casing must match the dashboard exactly.
- Melange version: **1** (assumed — same caveat as Model A; confirm at first SDK create).
- Input tensor (served, CONFIRMED — matches export exactly): `x` float32[1,3,48,320],
  NCHW, **BGR** channel order; normalize (pixel/255 − 0.5)/0.5 → range **[−1,1]**.
  Height fixed 48; each detected crop is aspect-preserving resized to height 48,
  width = min(round(48·w/h), 320), then **right-padded with zeros to width 320**
  (pad — never stretch).
- Output tensor (served, CONFIRMED — matches export exactly): `fetch_name_0`
  float32[1,40,838] — **time-major**: 40 CTC time-steps × 838 classes, **Softmax baked**
  (already probabilities). Dim layout: [batch=1, T=40, C=838].
- Post-processing baked into ONNX? **Softmax: YES (baked)** — apply NO extra softmax.
  **CTC greedy decode: NO** — pure Dart.
- Classes / labels: **838 CTC classes** (NOT 438 — do not copy any other app's
  dictionary, e.g. LiveDocRedact's): index **0 = CTC blank** (skip); indices
  **1..836** = `latin_charset.txt` lines 1..836 (836 chars, order preserved — includes
  accented Latin for FR/ES/DE/IT/PT); index **837 = space `' '`**. The dictionary ships
  in this folder as `latin_charset.txt` and becomes a Flutter asset; Dart prepends
  blank@0 and appends space@837 to rebuild the 838-class map.
- modelMode to use and why: **RUN_AUTO** — same rationale as Model A (CLAUDE.md §5: no
  client mode steers the served artifact; native-console served artifact is ground
  truth).
- GATE-0 benchmark (dashboard, informational): NPU min 0.51 / med 1.31 / avg 4.02 ms
  (~×51 vs CPU); GPU med 49.0 ms; CPU med 32.4 ms. Accuracy 13.25–29.18 dB (healthy —
  no anomaly on this model). Deployability 98%, FP32 100%. Size 2.07–8.22 MB across
  3 quantizations.

---

## Input source
- Rear camera, LIVE, cheapest usable pixel format (BGRA on iOS, YUV420 on Android →
  convert). Both models expect **BGR** channel order (PaddleOCR convention) — do NOT
  swap to RGB anywhere in the pipeline.
- Orientation handling required: measure the real buffer WxH on-device and show it on
  the HUD. On the PyroGuard iOS setup the BGRA buffer arrived UPRIGHT (720×1280) needing
  NO rotation — do not assume landscape; the historical bug was a *spurious* rotation,
  not a missing one. Android YUV420 buffers may differ — measure, don't assume.
- Scene text is arbitrary-orientation/perspective (angled street signs, tilted menus),
  so there are TWO orientation layers: (a) the frame/buffer orientation into the
  detector, and (b) the per-region **quad deskew** into the recognizer. DBNet finds the
  angled regions; Dart must perspective-warp each quad upright before the recognizer
  sees it. Per-crop deskew is even more critical here than for documents.

## Pre-processing pipeline (ordered, exact)

**Detector (per detection frame — see Latency budget for cadence):**
1. Capture frame bytes; convert source pixel format → BGR (drop alpha; keep BGR order —
   no RGB swap).
2. Letterbox-resize the frame to **736×736** preserving aspect (pad; record scale +
   pad offsets for the exact inverse). 736, NOT 640 — this detector is 736×736
   (divisible by 32, DBNet requirement).
3. Normalize per channel after /255: subtract mean [0.485,0.456,0.406], divide by std
   [0.229,0.224,0.225] (BGR channel order 0,1,2 as exported).
4. Reorder to NCHW [1,3,736,736]; flatten into a pre-allocated Float32List; wrap as
   `Tensor.float32List(data, shape: [1,3,736,736])`.

**Recognizer (per selected text region — see Latency budget for K/staggering):**
5. **Deskew:** perspective-warp the detected quad to an upright axis-aligned rectangle
   (quad → rect warp). Do not feed raw axis-aligned bounding boxes of angled quads.
6. Aspect-preserving resize the upright crop to height 48, width =
   min(round(48·w/h), 320).
7. Right-pad with zeros to width 320 → [3,48,320], BGR. Pad, never stretch.
8. Normalize (pixel/255 − 0.5)/0.5 → [−1,1]; NCHW [1,3,48,320]; wrap as
   `Tensor.float32List(data, shape: [1,3,48,320])`.

## Post-processing pipeline (ordered, exact)

**Detector output [1,1,736,736]** (probability map in [0,1] in 736×736 letterboxed
space; Sigmoid ALREADY baked — apply NO extra sigmoid):
1. Binarize at prob threshold **0.3**.
2. Find connected components / contours of the binary map.
3. For each candidate region: compute the mean probability inside; drop if
   < box_thresh **0.6**.
4. Fit a minimum-area rotated box; **unclip** (dilate) by unclip_ratio **1.5** to
   recover full glyph extents (DB shrinks text kernels during training).
5. Undo the letterbox (exact reverse of pre-proc step 2: subtract pad offsets, divide by
   scale) to map each quad into screen/frame space.
6. Emit an ordered list of text-region quads (reading order: top→bottom, left→right).

**Recognizer output [1,40,838] per crop** (time-major; Softmax ALREADY baked — apply NO
extra softmax):
7. For each of the 40 time-steps: argmax over the 838 classes → index sequence of
   length 40. (Argmax runs over the LAST axis, C=838 — not over T.)
8. Collapse consecutive duplicate indices (CTC merge).
9. Drop blanks (index 0).
10. Map each remaining index → char: 1..836 → `latin_charset.txt`[idx−1]; 837 → `' '`.
11. Confidence = mean of the per-step max prob over the kept (non-blank) steps.
12. Emit `RecognizedText{ quad, string, conf }`. (Optional downstream: local translate
    of `string` — no model, out of scope for v1 unless trivial.)

## Latency budget & frame scheduling (MANDATED)

Plan for the **CPU fallback** as the realistic default (CLAUDE.md §5: "benchmarked ≠
served"): detector ~**169 ms** + recognizer ~**32 ms PER CROP** (dashboard CPU medians).
A scene frame typically has ~1–10 text regions (signs/menus — fewer than a document),
but an unbudgeted frame with 10 regions is ~169 + 10×32 ≈ 490 ms on CPU. If the NPU path
is served (3.8 ms det / 0.5 ms rec), the same frame is real-time — but that must be
confirmed on the device console, not assumed.

The worker MUST implement all of the following (not optional):
1. **Top-K recognition per frame:** recognize at most K regions per frame (default
   K = 3, tunable), prioritized by region area (largest/nearest signs first) or by
   cache-miss status.
2. **Staggered recognition with IoU-keyed caching:** key each recognized string to its
   quad; on subsequent frames, match new quads to cached results by IoU (e.g. ≥ 0.5
   after motion) and re-display the cached string WITHOUT re-running the recognizer.
   Only cache-miss / changed regions consume recognizer budget. Evict stale entries
   (not matched for N frames).
3. **Detection cadence:** run the detector every frame ONLY if measured latency allows;
   otherwise every Nth frame (N adaptive to measured detector ms), re-drawing cached
   overlays (optionally motion-shifted) between detection frames so the overlay still
   tracks as the camera moves.
4. **`_busy` guard / frame dropping:** never queue frames; drop while inference is in
   flight (VALIDATION.md Tier B).
5. **HUD latency readouts:** detector ms, recognizer ms-per-crop, and crops-run-this-
   frame — visible on-screen (release builds cannot log from Dart; the HUD is the only
   observability).

## UI
- Visual design = worker's choice. Functional must-haves only:
  - **Live text overlay pinned to regions:** decoded string + confidence anchored to
    each detected quad, tracking the regions live as the camera moves (via the IoU
    cache between detection frames).
  - **Latency HUD:** detector ms + recognizer ms (per crop and/or total), plus buffer
    WxH and detector heatmap min/max/mean (the anomaly check) on a debug HUD line.
  - **Offline badge:** a clear "works offline / no signal needed" indicator — this is
    the demo's whole pitch for the traveler persona.
  - Live count of regions read.
  - OPTIONAL translate toggle only if trivially implementable in pure Dart; otherwise
    omit from v1 UI entirely (no model exists for it).
- Product display name **GlyphGo** applied per the display-name rules above; custom
  launcher icon (1024×1024 `assets/icon/app_icon.png` via `flutter_launcher_icons`,
  `remove_alpha_ios: true`) with a sign/glyph/travel motif (CLAUDE.md §4 binding).

## Platform targets
- **iOS 16.6+**, **Android minSdk 24**.
- Known OS traps (CLAUDE.md §5, all binding):
  - **GPU/MPSGraph history:** an FP32-GPU CoreML artifact can load cleanly then SIGABRT
    at first inference in Apple's MPSGraph compiler on iOS/macOS 26.3+; uncatchable in
    Dart; NOT client-fixable (no modelMode avoids it — all four modes return the same
    artifact). If hit, escalate to ZETIC to filter GPU server-side for that OS. Note
    the detector's GPU median here is 219 ms anyway — GPU is not the desirable path.
  - **Read the SERVED artifact from the native console**
    (`xcrun devicectl device process launch --console --terminate-existing --device
    <UDID> <bundleId>`); the served target+apType is ground truth, not the requested
    mode. Filtering GPU can drop to CPU, not NPU — "no crash" and "NPU speed" are
    separate wins.
  - **Release builds on device** (debug hangs on recent iOS/Xcode; debug icons don't
    launch standalone). Dart `print`/`debugPrint` does NOT reach the console in
    release — all diagnostics go on the UI/HUD.
  - **First-launch double download:** TWO models download on first launch over the
    network; on poor conference Wi-Fi that is two spinners. Pre-download/pre-warm and
    rehearse a fresh install. Ensure both models are cached and not re-downloaded per
    launch; warm EACH model with one dummy inference after load.
  - **Non-determinism acceptance:** server-side selection can serve a different
    artifact minute-to-minute. Acceptance = clean runs across multiple cold starts and
    at least one fresh install; re-verify after any backend/model re-target.
  - iOS simulator is a dead end (device-only xcframework slice, no camera) — every
    iteration is a signed device build.

## Validation focus (Tier A traps for THESE models — each needs a hand-built-data test)
- **CTC charset off-by-one (the #1 silent-wrong trap):** assert blank = index 0 is
  skipped, index i (1..836) maps to `latin_charset.txt`[i−1], and index 837 → space.
  Assert the class count is exactly **838 — NOT 438**; do not copy LiveDocRedact's (or
  any other app's) dictionary or class count. An off-by-one shifts every character.
- **Time-major [1,40,838] decode:** hand-build a logit tensor where a known step
  sequence spells a known short word; assert stepping is over T=40 with argmax over the
  LAST axis (C=838), and that CTC merge + blank-strip yields the exact string. A
  transposed read produces plausible garbage.
- **Pad-not-stretch to W=320:** a narrow crop must be aspect-resized to height 48 then
  right-padded with zeros — assert the pad region decodes to blank steps (padding must
  not be read as characters) and that no stretch path exists.
- **Letterbox(736) inverse round-trip:** forward-letterbox a known quad to 736×736,
  inverse back, assert return within tolerance; inverse must be the exact reverse order
  of the forward steps. 736, not 640.
- **No extra activation:** Sigmoid (detector) and Softmax (recognizer) are BOTH baked —
  assert Dart applies neither. Test against realistically-ranged tensors (probs already
  in [0,1] / rows summing to 1).
- **BGR assertion:** both models take BGR; assert the pre-proc feeds BGR (a known
  colored test pixel lands in the expected channel plane). An RGB swap silently
  degrades accuracy.
- **Dual orientation:** (a) frame/buffer orientation — assert the chosen transform
  round-trips a known quad for the buffer orientation actually measured on-device (HUD
  WxH + one raw quad); (b) crop deskew — a perspective-warped (deskewed) angled crop
  must decode to the same string as its axis-aligned upright version.
- **Threshold boundaries:** prob-threshold 0.3 (just-below pixel dropped, just-above
  kept) and box_thresh 0.6 mean-probability boundary for a region; unclip_ratio 1.5
  grows a known box by the expected amount.
- **Budget-scheduler tests:** top-K selection picks the K largest/priority regions;
  IoU-keyed cache hits (quad moved slightly → cached string reused, recognizer NOT
  invoked) and misses (new/changed region → recognizer invoked); stale-entry eviction;
  detection-cadence logic (Nth-frame gating) exercised with a fake clock.
- **Coordinate spaces:** detector map lives in 736×736 letterboxed space; recognizer
  crop in its own 48×320 padded space; screen-space quads only after the letterbox
  inverse. Tests must pin each space with known values.
- **Tier C carry-over (surface at GATE 3, not testable off-device):** the detector's
  anomalous −4.12–0.00 dB dashboard accuracy row — on-device, verify the served
  detector returns a real non-degenerate heatmap (HUD min/max/mean; text regions light
  up on a known sign) before trusting any downstream result; plus the standard Tier C
  checklist (served artifact/console readout, modelMode honesty, signing/OS gates,
  release build, double first-launch download, non-determinism acceptance, embedded
  personal key).

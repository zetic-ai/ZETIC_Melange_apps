# SPEC: LiveGate  (Stage-0 Explorer stub — GATE-0 late-binding fields marked)

> Pre-drafted by the Explorer. Everything the two ONNX files reveal is filled in and
> locally verified in onnxruntime. ONLY the Melange name/version fields are late-binding,
> marked **[LATE-BINDING — placeholder until GATE-0 paste-back]**. The orchestrator
> finalizes this after the local ONNX contract is set (does not wait for the upload).
> Suggested user-facing product name (worker sets at GATE 3): **"LiveGate"** (or a
> punchier variant e.g. **"GateKeeper" / "TrueFace"**) — folder/bundle-id stay `LiveGate`,
> Melange names stay `ajayshah/LiveGatePAD` + `ajayshah/LiveGateFace`.

## One-line pitch
A fully on-device KYC gate: point the front camera at a live selfie; the app decides
**LIVE vs SPOOF** (rejects a printed photo or a phone-screen replay) and, for a real
face, returns a **1:1 match score** against an enrolled reference — no face bytes ever
leave the device. For identity-verification / KYC vendors who cannot stream biometrics
to a cloud.

## Pipeline architecture (binding — this is the integration design, not the worker's to invent)

```
 front-camera frame (BGRA iOS / YUV420 Android)  ─┐
        │                                          │  (enrollment: one-time, same path,
        ▼                                          │   store the 128-d reference vector)
 ┌────────────────────────────┐   FACE DETECT (app-side, in-repo)
 │ 0. FACE DETECT + LANDMARKS  │   google/MediaPipe-Face-Detection (BlazeFace) → bbox
 └────────────────────────────┘   google/MediaPipe-Face-Landmark → 5 pts (eyes/nose/mouth)
        │                                          │
        ├──────────────► PAD branch                └──────────────► FACE branch
        │  crop = 2.7× margin around bbox                │  align = 5-pt affine warp to
        │  → resize 80×80, BGR, RANGE [0,255]            │    ArcFace template, 112×112,
        ▼                                                ▼    BGR, RANGE [0,255]
 ┌────────────────────────────┐  ajayshah/LiveGatePAD  ┌────────────────────────────┐  ajayshah/LiveGateFace
 │ 1. PAD (MiniFASNet-V2)      │  in[1,3,80,80]         │ 2. FACE EMBED (SFace)       │  data[1,3,112,112]
 │                             │  → out[1,3]            │                             │  → fc1[1,128]
 └────────────────────────────┘                        └────────────────────────────┘
        │  softmax; LIVE = p[1]                                │  L2-normalize → cosine vs
        ▼  LIVE if p[1] ≥ 0.45 else SPOOF                      ▼  enrolled vector
   liveness verdict                                       match score; MATCH if cos ≥ 0.363
        └──────────────────────────┬──────────────────────────┘
                                    ▼
          UI VERDICT:  ✅ LIVE + MATCH  |  ✅ LIVE + NO-MATCH  |  🚫 SPOOF (gate fails)
```

**The two Melange models are single forward passes.** Face detection + landmark
alignment are app-side; the repo already ships both as Melange models
(`google/MediaPipe-Face-Detection` = BlazeFace bbox; `google/MediaPipe-Face-Landmark` =
mesh/landmarks for the 5-point ArcFace alignment). The gate logic (crop geometry,
softmax/threshold, L2-norm + cosine, enrollment vector storage) is pure-Dart, worker-owned.

---

## Model

### Model 1 — PAD / anti-spoof (liveness)
- Source (HF repo / origin): garciafido/minifasnet-v2-anti-spoofing-onnx (**Apache-2.0**),
  provenanced to minivision-ai Silent-Face-Anti-Spoofing `2.7_80x80_MiniFASNetV2.pth`.
- Architecture: **MiniFASNet-V2** (mobile silent-PAD; Conv/BN/PRelu/MatMul). ~1.76 MB.
- Melange model name: **[LATE-BINDING — placeholder until GATE-0 paste-back]** (requested: `ajayshah/LiveGatePAD`)
- Melange version: **[LATE-BINDING — placeholder until GATE-0 paste-back]** (requested: 1)
- Input tensor: float32 **[1,3,80,80]**, NCHW, **BGR**, **RANGE [0,255] — do NOT ÷255**
  (⚠️ contradicts the model card, which says /255; the ONNX SATURATES at [0,1]. Empirically
  verified — see model_selection.md). Fed a **2.7× face-margin crop** resized to 80×80.
- Output tensor: float32 **[1,3]** raw logits. Apply softmax. **LIVE = softmax[class 1]**;
  classes 0 & 2 are the two spoof types (⚠️ NOT the card's `[live,print,replay]` ordering).
- Post-processing baked into ONNX? **No.** softmax + threshold is pure-Dart.
- Classes / labels: `[spoof_a, LIVE, spoof_b]`; decision `LIVE if p[1] ≥ 0.45`.
- modelMode to use and why: **RUN_AUTO** (default). No client mode steers a crashing
  artifact; GPU/MPSGraph traps are handled server-side by ZETIC (CLAUDE.md §5).

### Model 2 — Face embedding (1:1 match)
- Source: opencv/face_recognition_sface (**Apache-2.0**, OpenCV Zoo SFace, arXiv:2205.12010).
  (InsightFace w600k_mbf and EdgeFace were REJECTED — both non-commercial; see model_selection.md.)
- Architecture: **SFace / MobileFaceNet-style ArcFace** (Conv/BN/PRelu/Gemm). ~38.55 MB.
- Melange model name: **[LATE-BINDING — placeholder until GATE-0 paste-back]** (requested: `ajayshah/LiveGateFace`)
- Melange version: **[LATE-BINDING — placeholder until GATE-0 paste-back]** (requested: 1)
- Input tensor: float32 **[1,3,112,112]**, NCHW, **BGR**, **RANGE [0,255]** — the
  `(x-127.5)/128` normalization is **baked into the graph** (Sub/Mul first ops), so feed
  raw 0..255 pixels. Fed a **5-point-aligned** 112×112 face crop.
- Output tensor: float32 **[1,128]** face embedding. **NOT L2-normalized in-graph** —
  L2-normalize in Dart before cosine.
- Post-processing baked into ONNX? **No.** L2-norm + cosine vs enrolled vector is pure-Dart.
- Classes / labels: N/A (128-d embedding). Decision `MATCH if cosine ≥ 0.363` (0.363–0.40
  recommended; higher thr → lower impostor-accept for a stricter KYC gate).
- modelMode to use and why: **RUN_AUTO** (default).

---

## Input source
- **Front (selfie) camera**, cheapest usable pixel format (BGRA on iOS, YUV420 on Android → BGR).
- Both models expect **BGR** channel order and **[0,255]** range (no normalization in Dart
  for either — PAD needs raw [0,255]; SFace normalizes in-graph).
- Orientation handling: measure the real buffer WxH on-device; on the PyroGuard iOS setup the
  BGRA buffer arrived **upright (720×1280)** needing **NO** rotation. Do NOT assume landscape —
  the real PyroGuard bug was a *spurious* 90° rotation. Face crops are orientation-sensitive.

## Pre-processing pipeline (ordered, exact)
**Per frame (shared):**
1. Capture frame → convert source pixel format to **BGR** (drop alpha; YUV→BGR on Android).
2. Run face detect (`google/MediaPipe-Face-Detection`) → best face **bbox** (x,y,w,h).
   If no face, show "no face" and skip both models.
3. Run landmarks (`google/MediaPipe-Face-Landmark`) → 5 key points (both eye centers,
   nose tip, both mouth corners) for the FACE alignment.

**PAD branch:**
4. Compute the **2.7× margin box** around the bbox center (MiniVision `CropImage` geometry —
   see the exact recipe below), clip-shift to image bounds.
5. Crop from the ORIGINAL frame, resize to **80×80** (bilinear), keep **BGR**, keep **[0,255]**
   (⚠️ do NOT ÷255), reorder NCHW → `Tensor.float32List(data, shape:[1,3,80,80])`.

**FACE branch:**
6. Estimate the similarity/affine transform from the 5 detected points to the standard
   ArcFace 112×112 template (the 5 canonical points), warp the ORIGINAL frame to **112×112**.
7. Keep **BGR**, keep **[0,255]** (SFace normalizes in-graph), reorder NCHW →
   `Tensor.float32List(data, shape:[1,3,112,112])`.

### Exact 2.7× crop recipe (PAD — load-bearing; port verbatim to Dart)
```
scale = min((H-1)/bh, (W-1)/bw, 2.7)          # H,W = frame dims; bx,by,bw,bh = bbox
nw, nh = bw*scale, bh*scale
cx, cy = bx + bw/2, by + bh/2
x1, y1 = cx - nw/2, cy - nh/2
x2, y2 = cx + nw/2, cy + nh/2
# clip-shift so the box stays fully inside the frame (shift, don't just clamp):
if x1<0: x2-=x1; x1=0
if y1<0: y2-=y1; y1=0
if x2>W-1: x1-=(x2-W+1); x2=W-1
if y2>H-1: y1-=(y2-H+1); y2=H-1
crop = frame[int(y1):int(y2)+1, int(x1):int(x2)+1] ; resize -> 80x80
```

## Post-processing pipeline (ordered, exact)
**PAD:**
1. Read `output` [1,3] raw logits → softmax over the 3 classes.
2. `liveScore = softmax[1]`; **LIVE if liveScore ≥ 0.45**, else SPOOF (gate fails).

**FACE:**
3. Read `fc1` [1,128]; **L2-normalize** (divide by ‖v‖).
4. `cosine = dot(normalized_probe, normalized_enrolled)` (enrolled vector was L2-normalized
   and stored at enrollment). **MATCH if cosine ≥ 0.363.**
5. Enrollment path: same detect→align→embed→L2-normalize, store the 128-d vector locally
   (Keychain/secure storage). No raw face image persisted.

**Gate decision (pure-Dart):** `LIVE && MATCH → PASS`; `LIVE && !MATCH → face-mismatch`;
`SPOOF → reject` (do NOT even show the match score for a spoof).

## UI
- Left to the worker. Functional must-haves: live front-camera preview with the detected
  face box; a big **LIVE / SPOOF** verdict; the **match score** (cosine) with a PASS/FAIL
  badge; an **enroll** action to capture the reference face; a visible **"on-device · no
  cloud"** badge (airplane-mode-friendly); an inference-latency / per-stage HUD (shown
  on-screen, since Dart `print` won't reach the release device console — CLAUDE.md §5).

## Platform targets
- iOS 16.6+, Android minSdk 24 (PyroGuard baseline).
- Known OS traps: (a) iOS/macOS 26.3+ CoreML-GPU MPSGraph crash — handled server-side by
  ZETIC; confirm the *served* artifact isn't GPU on affected OS via the device console.
  (b) "Benchmarked ≠ served" — budget CPU-speed until `runtimeApType=NPU` is confirmed.
  (c) **Up to FOUR models loaded** (2 MediaPipe + PAD + FACE) → cold-start = 4 loads;
  warm each with a dummy inference right after load, and `_busy`-guard / throttle frames.

## Validation focus (Tier-A traps most likely for THESE models — worker must cover with tests)
- **PAD input range (#1 silent-wrong):** the ONNX needs **BGR [0,255]**; ÷255 SATURATES it
  to a constant class-2 output (looks alive but is dead). Test: feed a hand-built [0,255]
  tensor and a ÷255 tensor and assert they differ (the ÷255 path is the bug).
- **PAD live-class index:** LIVE = softmax **[1]**, not [0]. Test a hand-built [1,3] logit
  vector → assert the decoder reads index 1 as liveness.
- **PAD 2.7× crop geometry:** round-trip a known bbox through the crop recipe and assert the
  clip-SHIFT behavior (shift the box in-bounds, don't just clamp one edge) reproduces the
  reference box; a tight/wrong-scale crop collapses the model (validated).
- **FACE channel order + range:** BGR, [0,255] (normalization is in-graph). A silent R/B
  swap or an errant ÷255 degrades cosine without throwing. Test the preprocessor keeps BGR
  [0,255].
- **FACE L2-normalization:** the embedding is NOT unit-norm from the graph; cosine requires
  normalizing BOTH vectors. Test: un-normalized dot ≠ cosine; normalized dot ∈ [-1,1] and a
  vector with itself → 1.0.
- **FACE 5-point alignment:** an unaligned / center-cropped face lowers cosine materially
  (94.8% crude vs ~99% aligned). Assert the affine warp maps the 5 template points correctly.
- **Threshold boundaries:** PAD `liveScore` at 0.45 and FACE `cosine` at 0.363 — just-below
  rejected, just-above accepted.
- **Orientation:** measure the real front-camera buffer WxH on-device; do not assume
  landscape (PyroGuard's bug was a *spurious* rotation). A sideways face → garbage crops.
- **Gate composition:** a SPOOF must fail the gate regardless of match score; assert the
  Dart gate never returns PASS when PAD says SPOOF.

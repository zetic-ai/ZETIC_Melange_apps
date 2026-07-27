# SPEC: VisualSearch (two-model pipeline)

> Stage-0 pre-drafted spec stub (Explorer). Everything the locally-verified ONNX
> contracts reveal is concrete; only the two Melange registered name/version fields are
> `[LATE-BINDING — placeholder until GATE-0 paste-back]`. Product/display name is the
> worker's to finalize; a suggested name is noted under UI.

## One-line pitch
Snap one photo of any product/object and get an instant, fully **on-device** visual
match — a YOLO11n detector localizes the salient item, a MobileCLIP2-S0 image tower
turns the crop into a compact 512-d vector, and ONLY that vector would hit the server
for catalog matching. The demo story: remove the per-tap server-GPU embedding call and
show an instant/offline result for fashion e-commerce (the "Musinsa wedge").

## Pipeline overview
`camera frame → [Model 1: DETECT] salient box → crop (+margin) → [Model 2: EMBED] 256×256 → 512-d unit vector → (server: nearest-neighbor over catalog vectors)`
On-device demo can also show local NN against a small bundled gallery to prove the loop without a server.

---

## Model 1 — DETECTOR
- Source (HF/origin): Ultralytics **YOLO11n**, COCO-pretrained (via `ultralytics` assets).
- Architecture: YOLO11 nano (anchor-free, single-stage), 2.6M params.
- Melange model name:  `[LATE-BINDING — placeholder until GATE-0 paste-back]` (proposed `ajayshah/VisualSearchDetect`)
- Melange version:     `[LATE-BINDING — placeholder until GATE-0 paste-back]` (first upload = 1)
- Input tensor: `images` float32[1,3,640,640], **NCHW**, **RGB**, letterboxed (pad value 114), values **0.0–1.0** (pixel/255).
- Output tensor: `output0` float32[1,84,8400], **channel-major**. Per anchor (over 8400):
  `[cx, cy, w, h, s_0 … s_79]` — 4 box params + 80 COCO class scores; coords in 640×640
  letterbox space. Stride across anchors, not across the 84 (see Validation focus).
- Post-processing baked into ONNX? **NO.** Threshold → cxcywh→xyxy → un-letterbox → NMS are pure-Dart.
- Classes / labels: 80 COCO. **The app uses the box purely as a salient-object localizer**
  (crop the top box); the class label is incidental/for the HUD only.
- Primary-box rule: take the **highest-confidence** box (ties → largest area). If **no**
  box clears the threshold (~3–5% of real photos, e.g. flatlays), **fall back to a
  center-crop (or whole frame)** for the embed step — never skip the embed.
- modelMode to use and why: **RUN_AUTO** (CLAUDE.md §5 — no client mode steers off a
  crashing artifact; GPU/MPSGraph issues are ZETIC-server-filtered).

## Model 2 — EMBEDDING (image tower only)
- Source (HF/origin): **MobileCLIP2-S0** image tower — `timm: fastvit_mci0.apple_mclip2_dfndr2b`. Text tower is **out of scope**.
- Architecture: FastViT (MCi0) conv+MHSA hybrid, 11.4M params, with the CLIP image
  projection head inside the model; **L2-normalization added inside the ONNX graph**.
- Melange model name:  `[LATE-BINDING — placeholder until GATE-0 paste-back]` (proposed `ajayshah/VisualSearchEmbed`)
- Melange version:     `[LATE-BINDING — placeholder until GATE-0 paste-back]` (first upload = 1)
- Input tensor: `image` float32[1,3,256,256], **NCHW**, **RGB**, values **0.0–1.0**
  (pixel/255), **NO ImageNet mean/std** (MobileCLIP uses plain [0,1], mean 0 / std 1).
- Output tensor: `embedding` float32[1,512], **already L2-normalized** (unit vector).
- Post-processing baked into ONNX? **YES** — projection + L2-norm are in-graph, so cosine
  similarity between two embeddings is a **plain dot product** (no Dart post-processing).
- Embedding dim: **512**. L2-norm in-graph: **YES**.
- ⚠ License: **apple-amlr** (Apple ML Research) — non-standard; fine for the demo, review
  before commercial productization (see model_selection.md).
- modelMode: **RUN_AUTO**. Note the FastViT self-attention heads are the ViT-style
  attention family implicated in the iOS-26 CoreML-GPU MPSGraph crash — watch served
  apType on iOS 26.3+; escalate to ZETIC to filter GPU if first inference aborts.

---

## Input source
- Rear camera, single-tap capture (still frame is enough; the demo is snap-then-search,
  not continuous streaming — though a live viewfinder is fine).
- Cheapest usable pixel format (BGRA on iOS / YUV420 on Android); convert to RGB.
- Orientation: **measure the real buffer WxH on-device** — do not assume landscape.
  PyroGuard's iOS buffer arrived upright (720×1280); the bug was a *spurious* overlay
  rotation. Confirm and only rotate if the measured buffer requires it.

## Pre-processing pipeline (ordered, exact)
**Stage A — detector input**
1. Capture frame bytes → RGB.
2. Letterbox-resize the full frame to 640×640 (pad 114), preserving aspect; record scale `r` and pad `(dx,dy)`.
3. Normalize /255.0 → [0,1].
4. Reorder to NCHW [1,3,640,640]; flatten to Float32List; `Tensor.float32List`.

**Stage B — embed input (after detect + crop)**
5. From the primary box (top-confidence), map back to original-frame pixels (inverse of step 2), clamp to bounds, optionally expand ~5–8% margin.
6. Crop the original RGB frame to that box (fallback: center crop / whole frame if no box).
7. Resize the crop to **256×256** (bicubic/bilinear), /255.0 → [0,1], **no mean/std**.
8. Reorder to NCHW [1,3,256,256]; flatten; `Tensor.float32List`.

## Post-processing pipeline (ordered, exact)
**Detector**
1. Read `output0` [1,84,8400] as **channel-major**: for anchor a, score_c = out[4+c][a].
2. Keep anchors where max class score > threshold (default 0.25).
3. cxcywh → x1y1x2y2 (still in 640 letterbox space).
4. **Un-letterbox**: `x = (x_lb - dx)/r`, `y = (y_lb - dy)/r` (exact reverse of pre-step 2), clamp to frame.
5. **Global NMS** (IoU 0.5) — this is a class-agnostic salient-object localizer, so
   global NMS is correct (NOT per-class). Take the top box as primary.

**Embedding**
6. Read `embedding` [1,512] — it is already unit-norm; use directly.
7. Similarity = **dot product** with each catalog/gallery vector; rank descending; top-K are the matches.
8. (No sigmoid, no softmax, no normalization needed on-device.)

## UI
- Left to the worker. Functional must-haves: live camera / snap button; draw the primary
  detector box with confidence; show the crop that gets embedded; show the top-K nearest
  matches from a bundled demo gallery with cosine scores; inference-latency readout
  (detect ms / embed ms). Suggested product name: **SnapSeek** (or worker's choice) —
  distinct from folder/model names, set as iOS `CFBundleDisplayName` / Android
  `android:label` / in-app title.

## Platform targets
- iOS 16.6+, Android minSdk 24 (match repo baseline).
- Known OS traps: FP32-GPU CoreML crash in MPSGraph on iOS/macOS 26.3+ for ViT-style
  attention — the embedding tower is a FastViT hybrid (has MHSA), so it is a candidate.
  Read the *served* target+apType from the native console; not client-fixable (ZETIC
  filters GPU server-side). Realistic non-crash fallback is CPU (TFLITE_FP16), not NPU.

## Validation focus (the correctness traps most likely for THIS pipeline)
- **Channel-major decode** of [1,84,8400] (stride across 8400 anchors, not across 84) —
  hand-built one-anchor tensor with a known box+class, assert correct read.
- **Letterbox inverse round-trip** — forward-letterbox a known box then invert; must
  return to the original within tolerance (exact reverse order).
- **Coordinate space** — 640 letterbox space vs original-frame pixels; every box shifts if wrong.
- **Crop→embed mapping** — the box used for cropping must be in *original-frame* pixels,
  not letterbox space; assert a known box crops the right region.
- **Global (not per-class) NMS** for the salient-object localizer.
- **Threshold boundary** — just-below dropped, just-above kept (default 0.25).
- **Embedding preprocessing** — plain /255 with **no ImageNet mean/std** (a wrong mean/std
  silently degrades retrieval); assert a fixed input → expected 512-d unit vector
  (‖v‖≈1) and that cosine(v,v)=1.
- **No-detection fallback** — assert the pipeline still embeds (center-crop) when the
  detector returns zero boxes.
- **Orientation** — assert the chosen transform round-trips a known box for the buffer
  orientation you believe you have; confirm real buffer WxH on-device.

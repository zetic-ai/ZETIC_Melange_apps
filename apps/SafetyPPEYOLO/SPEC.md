# SPEC: SafetyPPEYOLO — PRE-DRAFTED STUB (GATE-0 fields pending human paste-back)

Status: Stage-0 stub. Everything the ONNX reveals is filled in. Fields marked
`<GATE-0>` are unknowable until the human registers the model on the Melange
dashboard and pastes back the served values.

## One-line pitch
Real-time worker-safety PPE compliance detector (helmet / vest, worn vs missing)
for industrial site-safety and CCTV prospects — live booth demo pointing a phone
at people.

## Model
- Source (HF repo / origin): ayushgupta7777/safetyvision-yolov8, file `v2/best.pt`
- Architecture: YOLOv8s fine-tuned (13 PPE classes, 11.1M params)
- Melange model name: `<GATE-0 — register as ajayshah/SafetyPPEYOLO; confirm exact casing from dashboard>`
- Melange version: `<GATE-0 — expected 1>`
- Input tensor: float32[1,3,640,640], NCHW, RGB, values 0.0-1.0 (divide by 255)
- Output tensor: float32[1,17,8400]; per anchor [cx, cy, w, h, s0..s12];
  CHANNEL-MAJOR (stride across the 8400 anchors, not across the 17); coords in
  640x640 letterbox space; 8400 anchors across 80/40/20 grids
- Served input/output shapes: `<GATE-0 — paste from dashboard>`
- Post-processing baked into ONNX? NO NMS baked in. Class scores ARE already
  sigmoid-applied (do not re-apply sigmoid).
- Classes / labels (id order):
  ["Fall-Detected", "Gloves", "Goggles", "Hardhat", "Mask", "NO-Gloves",
   "NO-Goggles", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "No_Harness",
   "Person", "Safety Vest"]
  - DEMO CLASSES (render these): 3 Hardhat, 7 NO-Hardhat, 9 NO-Safety Vest,
    12 Safety Vest. Optionally 4 Mask / 8 NO-Mask as extras.
  - **Class 11 Person is DEGENERATE — measured 0 predictions on a 40-image GT set
    even at conf 0.05. The UI must not depend on person boxes.** Ignore ids
    0,1,2,5,6,10 too (weak/irrelevant for the demo); filter to the demo-class
    whitelist in the postprocessor.
- modelMode to use and why: RUN_AUTO `<GATE-0 — confirm>`. (Per CLAUDE.md §5: no
  client mode steers off a crashing artifact; the iOS 26.3+ GPU crash is handled
  server-side by ZETIC.)

## Input source
- Rear camera, cheapest usable pixel format
- Device held portrait. On the PyroGuard iOS setup the BGRA buffer arrived
  UPRIGHT (buf=720x1280) and the bug was a SPURIOUS 90° overlay rotation, not a
  missing one. Verify actual buffer WxH per device/format on a HUD debug line —
  measure, don't assume. (Android YUV420 may differ.)

## Pre-processing pipeline (ordered, exact — mirrors the validated eval harness)
1. Capture frame bytes (BGRA on iOS / YUV420 on Android)
2. Convert to RGB
3. Letterbox-resize to 640x640 preserving aspect: scale r = min(640/w, 640/h),
   bilinear resize to (round(w*r), round(h*r)), center on a gray canvas
   (pad value 0.5 after normalization, i.e. 127.5/255), pad offsets
   dx = (640-nw)//2, dy = (640-nh)//2
4. Normalize /255.0
5. Reorder to NCHW [1,3,640,640]
6. Flatten to Float32List, wrap as Tensor.float32List

## Post-processing pipeline (ordered, exact)
1. Read output float32[1,17,8400] CHANNEL-MAJOR: value(ch, a) = out[ch*8400 + a]
2. For each anchor a: cls = argmax over s0..s12, score = max (NO sigmoid — already
   applied). Threshold FIRST, then geometry (cheap rejection).
3. Per-class confidence thresholds: Hardhat 0.25; Safety Vest / NO-Hardhat /
   NO-Safety Vest 0.15 (measured: vest recall 0.26→0.35 at P 0.94 when dropping
   0.25→0.15). Drop non-whitelisted classes.
4. cxcywh -> x1y1x2y2 (still in 640-letterbox space)
5. Undo letterbox — exact inverse of pre-processing: x' = (x - dx) / r,
   y' = (y - dy) / r; clamp to frame bounds
6. PER-CLASS NMS (not global), IoU 0.45 — overlapping Hardhat + Safety Vest on
   the same worker must BOTH survive
7. Emit Detection{bbox, label, conf}; color-code worn (green) vs violation (red)

## UI
- Worker's choice. Functional must-haves: live overlay of boxes + confidence,
  worn-vs-violation color coding, per-class live count (e.g. helmets/vests/
  violations in frame), inference latency readout, HUD debug line
  (buffer WxH + first raw box) toggleable for bring-up.

## Platform targets
- iOS 16.6+, Android minSdk 24
- Known OS traps: FP32-GPU CoreML artifact crashes in Apple MPSGraph on iOS/macOS
  26.3+ (server-side-filtered by ZETIC; read the SERVED artifact from the native
  console). Release builds on device; simulator is a dead end (device-only
  xcframework slice). Dart prints don't reach the device console in release —
  HUD diagnostics only.

## Validation focus (Tier A traps most likely for THIS model)
- Channel-major [1,17,8400] decode (hand-built tensor with one known box)
- Letterbox inverse round-trip within tolerance (pad 0.5, bilinear, //2 offsets)
- NO double-sigmoid on class scores (scores already 0-1)
- Per-class (not global) NMS: overlapping Hardhat + Safety Vest both survive
- Per-class threshold boundaries (0.25 helmet / 0.15 vest+violations):
  just-below dropped, just-above kept
- Class-whitelist filter: ids 0,1,2,5,6,10,11 never rendered (esp. Person=11)
- Orientation: assert the chosen transform round-trips a known box; verify real
  buffer WxH on-device via HUD

## Demo validation evidence (Stage 0, measured)
- demo_validation/overlay_img_023.jpg — booth-range money shot: helmet 0.88 on
  hardhat wearer, no-helmet 0.87 on two bareheaded men, zero false boxes
- demo_validation/overlay_img_009.jpg — far-field: 4 helmets + 1 no-helmet, 0 FP
- demo_validation/overlay_img_027.jpg — vests 0.45-0.74 + helmet on roofers, 0 FP
- Full head-to-head numbers and weaknesses: model_selection.md

## Known model weaknesses the worker must design around
- Person class dead (see above) — never render or count it
- Far-field vest / violation recall is low; demo choreography should keep
  subjects within a few meters of the camera
- Best-lit, frontal poses score highest; a trade-show booth is the favorable case

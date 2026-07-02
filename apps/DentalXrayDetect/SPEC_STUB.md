# SPEC: DentalXrayDetect

> Pre-drafted Stage-0 stub. Everything the exported ONNX reveals is filled in.
> Fields marked `<<FILL AT GATE 0>>` await the human's Melange dashboard paste-back.

## One-line pitch
On-device dental radiograph analyzer that outlines caries, periapical lesions, and impacted
teeth on an x-ray directly in the operatory — no cloud upload, so radiographs/PHI never leave
the practice. Capability-proof wedge vs cloud-SaaS chairside dental AI (Overjet / VideaHealth).

## Model
- Source (HF repo / origin): liodon-ai/dental-panoramic-detector, weights `best.pt`
- Architecture: YOLO11n (Ultralytics), anchor-free detect head; trained on 9,928 dental
  panoramic radiographs (DENTEX + OralXrays-9). 2.58 M params, 6.3 GFLOPs, 10 MB ONNX.
- Melange model name: <<FILL AT GATE 0>>   (requested: DentalXRayDetect — registered under the
  ZETIC org; dashboard shows "ZETIC | DentalXRayDetect". Match case EXACTLY: capital R in "XRay",
  which differs from the folder name DentalXrayDetect.)
- Melange version: <<FILL AT GATE 0>>       (requested: 1)
- Input tensor: float32[1,3,640,640], NCHW, RGB order, value range 0.0–1.0 (pixels / 255)
  - Served input shape (echo from dashboard): <<FILL AT GATE 0>>
- Output tensor: float32[1,7,8400], channel-major. Per anchor: [cx, cy, w, h, s0, s1, s2] =
  4 box coords (PIXEL space in the 640×640 letterbox frame) + 3 class scores. 8400 anchors =
  80²+40²+20² across /8, /16, /32 strides.
  - Served output shape (echo from dashboard): <<FILL AT GATE 0>>
- Post-processing baked into ONNX? **NMS is NOT baked in.** Sigmoid **IS** baked in — class scores
  are already 0–1 (verified via onnxruntime on the exported graph); do NOT re-apply sigmoid in Dart.
  YOLO11 has no separate objectness channel — the 7 = 4 box + 3 class.
- Classes / labels (3, verified from checkpoint model.names):
  0 caries, 1 periapical_lesion, 2 impacted_tooth
- modelMode to use and why: RUN_AUTO (default). No client mode steers off a crashing artifact;
  the iOS-26 GPU/MPSGraph trap is handled server-side by ZETIC. See CLAUDE.md §5.

## Input source
- Primary: a still dental radiograph IMAGE (file/gallery pick, or a photo of the mounted x-ray).
  This is image-in analysis, NOT a live video pipeline like PyroGuard — single-shot inference on
  a selected radiograph is the natural chairside flow. (A live-camera mode is optional/secondary.)
- Pixel format: decode the picked image to RGB. Radiographs are effectively grayscale but the model
  expects 3 channels — replicate luma to RGB (or decode RGB directly).
- Orientation handling: for a file/gallery image, honor EXIF orientation; no camera-buffer rotation
  trap applies. If a live-camera mode is added, MEASURE the real buffer WxH before assuming rotation
  (see PyroGuard: the bug was a *spurious* rotation, not a missing one).

## Pre-processing pipeline (ordered, exact)
1. Load the radiograph image; apply EXIF orientation.
2. Convert to RGB (replicate grayscale luma to 3 channels if source is single-channel).
3. Letterbox-resize to 640×640 (pad value 114-gray / 0.447, preserve aspect ratio).
4. Normalize /255.0 → 0.0–1.0.
5. Reorder to NCHW [1,3,640,640].
6. Flatten to Float32List, wrap as Tensor.float32List(data, shape: [1,3,640,640]).

## Post-processing pipeline (ordered, exact)
1. Read output as channel-major [1,7,8400]: stride across the 8400 anchors, NOT across the 7.
2. For each anchor, class scores are channels 4..6; take max class score + its index.
3. Class scores are ALREADY sigmoid-activated (0–1) — do NOT re-apply sigmoid (double-activation bug).
4. Keep anchors where max class score > threshold. Use conf 0.45 (model card: at 0.25 caries
   over-fires on adjacent teeth). Threshold BEFORE box geometry.
5. cxcywh → x1y1x2y2 (coords are in 640×640 letterbox PIXEL space, not normalized).
6. Undo letterbox (exact reverse of pre-processing) into original-image space.
7. Per-class NMS, IoU 0.35 (model card recommendation; per-class, NOT global — a caries box and an
   impacted-tooth box may legitimately overlap).
8. Emit Detection{bbox, label, conf}.

## UI
- Left to the worker. Functional must-haves: overlay of detected regions + confidence on the
  radiograph, per-class count (caries / periapical_lesion / impacted_tooth), inference latency
  readout. Because output is a screening hint, surface a clear "research capability proof, not a
  diagnostic device" disclaimer in-app.

## Platform targets
- iOS minimum 16.6+, Android minSdk 24.
- Known OS traps for this model/artifact:
  - FP32-GPU CoreML artifact can crash in Apple MPSGraph on iOS/macOS 26.3+ (not client-fixable;
    ZETIC filters GPU server-side). Read the *served* target+apType from the native console.
  - 640×640 tiny YOLO11n → light compute; realistic non-crashing fallback is often TFLITE_FP16/CPU.

## Validation focus
- Channel-major [1,7,8400] decode (stride across anchors, not the 7) — easiest trap.
- Do NOT re-apply sigmoid — scores are already 0–1 in-graph (double-activation would crush scores).
- Letterbox inverse round-trip at 640 → original image space (assert a known box round-trips).
- Coordinate space: pixel-space 640, not normalized 0–1.
- Per-class (not global) NMS at IoU 0.35; conf threshold 0.45.
- Domain-shift caveat: weights are panoramic-trained; on bitewing/periapical inputs caries recall is
  limited (screening hint, not a count). This is a data caveat to disclose, not a code bug.
- License: CC-BY-NC-4.0 (non-commercial) — capability-proof only. See model_selection.md.

# 🔥 PyroGuard — Flutter

Real-time **fire & smoke detection** demo for [ZETIC.ai](https://zetic.ai), powered by the
**ZETIC Melange** Flutter SDK. A fine-tuned YOLO11s model runs fully on-device on the mobile
NPU — Melange picks the optimal accelerator binary per device automatically.

Built for industrial-safety scenarios (smart factories, warehouses, energy sites).

---

## ✨ Features

- Full-screen live camera feed with real-time bounding-box overlay.
- On-device YOLO11s inference via Melange (NPU/GPU/CPU auto-selected).
- Per-class live counts (🔥 fire / 💨 smoke) and inference latency HUD.
- Adjustable confidence threshold (bottom sheet, 0.10–0.80, default 0.25).
- Pure-Dart letterbox preprocessing + NMS post-processing.
- Dark, fire-coded product UI (accent `#FF4500`).

## 🚀 Quick start

```bash
cd apps/FireDetectionYOLO/Flutter
flutter pub get
flutter run            # on a physical iOS or Android device
```

> The camera + NPU pipeline requires a **physical device**. Simulators/emulators
> have no real camera and no NPU.

## 🧠 Model

| | |
|---|---|
| Architecture | YOLO11s, fine-tuned for fire/smoke |
| Melange model | `ajayshah/FireDetectionYOLO` (version 1) |
| Inference mode | `ModelMode.runAuto` (RUN_AUTO) |
| Input | `float32[1, 3, 640, 640]`, NCHW, normalized 0–1 |
| Output | `float32[1, 6, 8400]` → `[cx, cy, w, h, fire_conf, smoke_conf]` |
| Classes | `fire`, `smoke` |
| NMS | implemented in pure Dart (IoU threshold 0.45) |

The Melange personal key in `lib/services/melange_service.dart` is a ZETIC **dev** key for
this demo. Swap in your own from the [Melange console](https://mlange.zetic.ai) for production.

## 🔧 Pipeline

```
CameraImage ──▶ preprocess (letterbox 640², BGR/YUV→RGB, /255, NCHW)  [isolate]
            ──▶ Melange model.run([Tensor])                          [main, NPU]
            ──▶ postprocess (threshold, decode, un-letterbox, NMS)    [isolate]
            ──▶ List<Detection> ──▶ CustomPainter overlay
```

Pre/post-processing run in background isolates via `compute()`; the native `model.run`
stays on the main isolate (its handle is bound there, and the NPU pass is only a few ms).

## 📁 Structure

```
lib/
  main.dart                  # app entry + theme wiring
  theme.dart                 # PyroGuard palette / ThemeData
  models/detection.dart      # Detection data class
  services/
    melange_service.dart     # model init / run / teardown
    preprocessor.dart        # camera frame → Float32List (NCHW)
    postprocessor.dart       # raw output → List<Detection>
    nms.dart                 # pure-Dart IoU + NMS
  screens/
    loading_screen.dart      # branded loader, real download progress
    camera_screen.dart       # live feed + overlay + HUD + settings
  widgets/
    detection_overlay.dart   # CustomPainter for boxes
    hud_bar.dart             # top HUD (wordmark + latency)
    stats_bar.dart           # bottom per-class counts
```

## 📦 Requirements

- Flutter ≥ 3.35, Dart ≥ 3.11
- iOS 16.6+ (Melange SDK floor) · Android API 24+
- Packages: `zetic_mlange`, `camera`, `image`

## 🧪 Tests

```bash
flutter test    # covers IoU, NMS, and output decoding
```

---

Powered by **ZETIC Melange**.

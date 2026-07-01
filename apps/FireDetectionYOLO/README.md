# Fire & Smoke Detection (PyroGuard)

<div align="center">

**Real-Time On-Device Fire & Smoke Detection**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![Flutter](https://img.shields.io/badge/Framework-Flutter-blue.svg)](Flutter/)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](Flutter/android/)
[![iOS](https://img.shields.io/badge/Platform-iOS-lightgrey.svg)](Flutter/ios/)

</div>

> [!TIP]
> **View on Melange Dashboard**: [ajayshah/FireDetectionYOLO](https://mlange.zetic.ai/p/ajayshah/FireDetectionYOLO?tab=summary&version=1) — Contains generated source code & benchmark reports.

PyroGuard is a Flutter demo that runs a **YOLO11s fire/smoke detector** fully on-device via
**Melange**, streaming the live camera feed and overlaying real-time detection boxes with a
latency + detection-count HUD. The model runs on the device NPU (Apple Neural Engine / Qualcomm
Hexagon) — no cloud inference.

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure your key**: open `Flutter/lib/services/melange_service.dart` and replace
   `YOUR_MLANGE_KEY` with your Melange personal key:
   ```dart
   static const String _personalKey = 'YOUR_MLANGE_KEY';
   ```
3. **Run the App** (physical device required — the Melange SDK is device-only, and the app needs a
   camera):
   ```bash
   cd Flutter
   flutter pub get
   flutter run --release
   ```
   - **iOS**: needs an Apple developer team for signing (set it in Xcode → Runner → Signing).
   - **Android**: enable USB debugging; first launch downloads the model (keep Wi-Fi on).

## 📚 Resources

- **Melange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/ajayshah/FireDetectionYOLO?from=use-cases)
- **Melange Model Library**: [Model Library](https://mlange.zetic.ai/model-library)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Model**: YOLO11s Fire/Smoke Detector ([leeyunjai/yolo11-firedetect](https://huggingface.co/leeyunjai/yolo11-firedetect))
- **Task**: Object Detection — classes: `fire`, `smoke`
- **Melange Project**: [ajayshah/FireDetectionYOLO](https://mlange.zetic.ai/p/ajayshah/FireDetectionYOLO?from=use-cases)
- **Version**: 1
- **Input**: `float32[1, 3, 640, 640]` (RGB, letterboxed, normalized 0–1, NCHW)
- **Output**: `float32[1, 6, 8400]` (cx, cy, w, h, fire, smoke — channel-major)
- **Key Features**:
  - Real-time fire/smoke detection from the live camera
  - NPU-accelerated via Melange (Apple Neural Engine / Qualcomm Hexagon)
  - Single Flutter codebase for iOS and Android

## 📁 Directory Structure

```
FireDetectionYOLO/
├── export.py     # Exports the YOLO11s fire/smoke model to ONNX (for Melange upload)
└── Flutter/      # Flutter app (iOS + Android) using the Melange SDK
    └── lib/
        ├── screens/    # Loading + live camera screens
        ├── services/   # Melange lifecycle, preprocess → run → postprocess, NMS
        ├── widgets/    # Detection overlay + HUD
        └── models/     # Detection model
```

## 📝 Notes

- **Camera orientation** is handled per-platform: iOS delivers the buffer display-upright, Android
  delivers it in sensor orientation (landscape), so the preprocessor rotates the frame upright before
  inference — important for detection accuracy.
- The model is downloaded on first launch by name/version; no model file ships with the app.

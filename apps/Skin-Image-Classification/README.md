# Skin Image Classification

<div align="center">

| **On-Device Skin Lesion Classification** |
|:---:|
| <img src="../../res/screenshots/skin-classification.gif" width="200" alt="On-device skin-lesion image classification"> |

</div>

<div align="center">

**Classify a Skin Photo's Pixels — 100% On-Device, No Cloud**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](Android/)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](iOS/)

</div>

> [!TIP]
> **View on Melange Dashboard**: [realtonypark/Skin_Cancer-Image_Classification](https://melange.zetic.ai/p/realtonypark/Skin_Cancer-Image_Classification) — Contains generated source code & benchmark reports.

Skin Image Classification runs a real dermatology **vision model fully on-device** via ZETIC
Melange. A **ViT image classifier** (7 HAM10000 lesion types) reads a skin photo's actual
pixels on the device NPU/Neural Engine: photo → 224×224 RGB tensor → 7-class softmax → top
class + confidence + ranked distribution, shown with curated per-condition guidance. No pixels
ever leave the phone — built to show skin-image AI companies they can deploy their **own**
models on-device. Swapping in a client's model is a one-line `AppConfig` change. iOS is
SwiftUI, Android is Jetpack Compose.

> ⚠️ **Not a medical device.** For demonstration only — the UI always shows a non-dismissible
> disclaimer and avoids the word "diagnosis".

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
   This replaces the `YOUR_MLANGE_KEY` placeholder in `iOS/SkinImageClassification/App/AppConfig.swift`
   and `Android/.../skinclassifier/core/AppConfig.kt` with your personal access token.
3. **Run the App**:
   - **Android**: Open `Android/` in Android Studio and run on a physical device (NPU) — see
     the [Android README](Android/README.md).
   - **iOS**: Generate + open the project under `iOS/` (`xcodegen generate`), then run on a
     physical iPhone — see the [iOS README](iOS/README.md).

> A **physical device is required** — the Melange SDK targets device NPUs and ships no
> Simulator/emulator slice. First launch downloads/compiles the classifier, then caches it for
> fully-offline inference.

## 📚 Resources

- **Melange Dashboard**: [View Model & Reports](https://melange.zetic.ai/p/realtonypark/Skin_Cancer-Image_Classification)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)
- **Platform deep-dives**: [iOS README](iOS/README.md) · [Android README](Android/README.md)

## 📋 Model Details

- **Model**: `realtonypark/Skin_Cancer-Image_Classification`
- **Task**: Skin-lesion image classification — ViT-base-patch16-224, 7 HAM10000 classes
- **I/O contract**: `1×3×224×224` NCHW Float32, RGB, normalized to `[-1, 1]` → 7 logits → softmax
- **Key Features**:
  - Fully on-device inference via Melange (Apple Neural Engine / mobile NPU)
  - Inference in ~tens of milliseconds; classifier cached after first download
  - Confidence ring + ranked class distribution, with low-confidence flagging and curated,
    severity-aware guidance per condition

This application showcases a **ViT skin-lesion classifier** using **Melange**, running entirely
locally so no photos leave the device. Swapping in a client's own model is a one-line
`AppConfig` change (plus the labels in `SkinClass`).

> ⚠️ **Validate preprocessing before demoing.** The Melange-converted graph may expect a
> different layout (NHWC), channel order (BGR), or normalization than the PyTorch default —
> the #1 source of a "wrong prediction on device" bug. All three are tunable in
> `AppConfig.Preprocess`; see the platform READMEs for the validation procedure.

## 📁 Directory Structure

```
Skin-Image-Classification/
├── Android/      # Jetpack Compose implementation with Melange SDK — see Android/README.md
└── iOS/          # SwiftUI implementation (XcodeGen) with Melange SDK — see iOS/README.md
```

For platform-specific pipeline, build, and validation notes, see the
[**iOS README**](iOS/README.md) and [**Android README**](Android/README.md).
</content>
</invoke>

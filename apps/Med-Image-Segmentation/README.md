# Med Image Segmentation

**On-Device Cardiac Ultrasound Segmentation**

> [!TIP]
> **View on Melange Dashboard**: [realtonypark/EchoNet-DeepLab-v3](https://mlange.zetic.ai/p/realtonypark/EchoNet-DeepLab-v3) - Contains generated source code and benchmark reports.

Med Image Segmentation runs a cardiac ultrasound left-ventricle segmentation
pipeline locally with Melange. The Flutter app plays bundled echo clips, runs
the EchoNet DeepLabV3 model on-device, overlays the LV cavity mask, and reports
latency, IoU, memory, and device information during the benchmark.

The app is built with Flutter and includes Android and iOS platform projects.

## Quick Start

1. **Get your Melange API key**: [Sign up here](https://mlange.zetic.ai)
2. **Configure API key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the app**:
   ```bash
   cd apps/Med-Image-Segmentation/Flutter
   flutter pub get
   flutter run
   ```

> A physical device is required for the Melange runtime. The first launch
> downloads and caches the model; after that, inference runs locally.

## Resources

- **Melange Dashboard**: [EchoNet-DeepLab-v3](https://mlange.zetic.ai/p/realtonypark/EchoNet-DeepLab-v3)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## Model Details

- **Segmentation model**: `realtonypark/EchoNet-DeepLab-v3`
  - Cardiac ultrasound left-ventricle cavity segmentation.
- **Task**: Echo clip frame -> LV segmentation mask
- **Input**: `(1, 3, 112, 112)` float32 RGB
- **Output**: `(1, 1, 112, 112)` float32 logits
- **Key features**:
  - Fully on-device inference via Melange
  - Live LV mask overlay on bundled echo clips
  - Device benchmark HUD with latency, IoU, memory, and frame metrics

## Directory Structure

```
Med-Image-Segmentation/
`-- Flutter/      # Flutter implementation with Android and iOS platform apps
```

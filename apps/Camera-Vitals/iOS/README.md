# Camera Vitals — iOS (SwiftUI)

A modern SwiftUI app that measures **heart rate from the front camera, 100% on-device**,
running the EfficientPhys rPPG model on the Apple Neural Engine via
[ZETIC Melange](https://docs.zetic.ai). Built to show camera-vitals companies (e.g. CarePlix)
that they can deploy their own models on-device — no cloud, no data leaving the phone.

## How it works

```
Front camera 30fps → Vision face ROI (locked/smoothed) → 72×72 RGB crop
  → 181-frame ring buffer → standardize → ZeticMLange EfficientPhys (NPU)
  → 180-sample pulse waveform → cumsum → detrend → bandpass 0.75–2.5Hz → FFT peak → BPM
```

- **Sliding window**: inference runs ~once per second over a 6 s window; a single-flight `busy`
  gate drops windows under load so capture never stalls.
- **Signal quality** (spectral SNR) gates and smooths the displayed BPM (median + EMA), so a
  noisy window never flashes a wild number.

## Project layout

This iOS app lives under `apps/Camera-Vitals/iOS/`:

- `CameraVitals/` — the app source (App, Capture, Vision, Pipeline, Signal, State, Views,
  Components, Util).
- `project.yml` — XcodeGen spec (pins `ZeticMLangeiOS` exact **1.6.0**, iOS 16.6, front camera).
- `CameraVitals.xcodeproj` — generated project.

> Model-prep tooling (EfficientPhys → `efficientphys.pt2` export and the end-to-end numeric
> validation script) lives with the model on the Melange dashboard, not in this app folder.

## Build & run

### Generate + compile-check (headless, no device)

```bash
cd apps/Camera-Vitals/iOS
brew install xcodegen
xcodegen generate
xcodebuild -project CameraVitals.xcodeproj -scheme CameraVitals \
  -destination 'generic/platform=iOS' -derivedDataPath build \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO build
```

### Run on a physical iPhone (A11+ for the Neural Engine)

The ZeticMLange xcframework is **device-only** (no simulator slice), and the app needs a real
camera — so it must run on an iPhone, not the simulator:

1. `open CameraVitals.xcodeproj`
2. Select the **CameraVitals** target → Signing & Capabilities → choose your **Team** (sets a
   signing identity; bundle id `ai.zetic.demo.CameraVitals`).
3. Plug in an iPhone, select it, press **Run**.
4. First launch downloads + compiles the model for the ANE (progress shown), then grant camera
   access and hold still in good light.

## Verification status

| Check | Status |
| --- | --- |
| Swift compiles + links against real ZeticMLange 1.6.0 | ✅ `BUILD SUCCEEDED`, 0 warnings |
| Model I/O contract `[181,3,72,72] → [180,1]` | ✅ verified |
| End-to-end BPM recovery (exported `.pt2`, synthetic pulse 50–140 bpm) | ✅ &lt; 2 bpm error |
| Inference speed | ✅ ~211 ms/window on CPU (4 threads); 5× under the 1 s budget — ANE much faster |
| Live camera accuracy vs a reference pulse oximeter | ⏳ requires your device |

## Notes

- rPPG is lighting/motion sensitive — measure in even light, holding still; expect a ~6 s warm-up.
- Heart rate only by design (most robust). HRV/respiration are derivable from the same waveform
  but were intentionally left out of this demo.
- Not a medical device — for demonstration only.

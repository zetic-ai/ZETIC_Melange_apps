# Camera-Vitals — Android (Kotlin / Jetpack Compose)

The Android port of the on-device rPPG heart-rate demo, running the same EfficientPhys model on
the device's NPU/GPU via **ZeticMLange** (`com.zeticai.mlange:mlange:1.8.1`). Feature-for-feature
identical to the iOS app in `../ios`.

## Pipeline
```
CameraX front cam (RGBA) → ML Kit face ROI (locked/smoothed) → 72×72 RGB crop
  → 31-frame ring → standardize → ZeticMLange EfficientPhys (NPU)
  → 30-sample chunk → stitch into rolling 300-sample buffer
  → cumsum → detrend → bandpass 0.75–2.5 Hz → FFT peak → BPM
```
Same architecture as iOS: detect every 3rd frame, crop every frame, inference every 30 frames on a
separate thread with a single-flight busy gate, HR after ~5 s, harmonic-SNR quality badge, median+EMA
smoothing, ring reset on face loss, bitmaps recycled per frame.

## Build & run
The Melange model is already deployed on the dashboard as **version 2** (the 31-frame low-memory
build) — no upload needed.

1. Open the `android/` folder in **Android Studio** (Koala / Ladybug or newer).
2. On first open, let Android Studio set up the Gradle wrapper (it will use Gradle 8.9 from
   `gradle/wrapper/gradle-wrapper.properties`). If you have Gradle on your machine you can instead
   run `gradle wrapper` in `android/`.
3. Gradle sync (pulls ZeticMLange, CameraX, ML Kit, Compose).
4. Run on a **physical Android phone** (camera + NPU). Grant the camera permission; hold still in
   even light. HR appears after the ~5 s warm-up; the badge turns green on a clean pulse; the
   "Measure 30s" button runs a guided scan → result bottom sheet.

> A device is required (the emulator has no real camera and the Melange runtime targets device NPUs).

## Config
All tuning lives in `app/src/main/java/ai/zetic/demo/cameravitals/AppConfig.kt`
(model name/version/key, window sizes, HR band, cadence) — mirrors the iOS `AppConfig`.

## Verification status
| Check | Status |
|---|---|
| Signal DSP (FFT, bandpass, detrend, quality) ported 1:1 | ✅ compiled with `kotlinc` + numerically recovers HR (0 bpm error on synthetic 50–140 bpm) |
| Full app compile | ⏳ build in Android Studio (no Android SDK on the authoring machine) |
| Live on-device run | ⏳ requires your Android phone |

The DSP is the algorithmically risky part and is verified; the rest is a faithful structural port of
the working iOS app. `screenOrientation` is locked to portrait.

Not a medical device — for demonstration only.

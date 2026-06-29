# DermaScope — On-Device Skin Analysis (Android)

A 1:1 Android port of the iOS DermaScope demo: a real dermatology **vision model**
classifies a skin photo's actual pixels **fully on-device** through ZETIC Melange — no
cloud, no data leaving the phone.

- **Skin vision model** — a ViT image classifier
  (`realtonypark/Skin_Cancer-Image_Classification`, 7 HAM10000 lesion types) deployed on
  Melange, run via `ZeticMLangeModel`.

The pitch in one screen: *swap the model `name` for your own Melange model and the rest of
the app is unchanged — your dermatology model, on-device, no cloud.*

> This is the **classifier-only** build (matching what the iOS app actually ships). The
> results screen renders curated per-condition guidance text from `SkinClass`; there is no
> LLM step.

## Architecture

```
Photo (camera / library)
  └─ ImagePreprocessor   Bitmap → 224×224 RGB → Float[1,3,224,224] (NCHW, [-1,1])
       └─ SkinClassifier   ZeticMLangeModel.run(arrayOf(tensor)) → 7 logits → softmax
            └─ Classification (top class + confidence + ranked distribution)
                 └─ ResultsScreen   verdict card + distribution bars + curated guidance
```

The classifier loads on a background thread with download progress (cached after first run).
All Melange SDK calls are funnelled onto a single `MelangeRuntime` thread (the SDK forbids
concurrent init/run).

## Project layout

| Area | Files |
|------|-------|
| App / config | `MainActivity.kt`, `core/AppConfig.kt` (key, model name + version, **preprocess config**), `ui/Theme.kt` |
| Core | `core/MelangeKit.kt` (load + tensor/softmax helpers), `core/MelangeRuntime.kt`, `core/ModelStatus.kt` |
| Models | `model/SkinClass.kt` (7 classes + severity + curated guidance), `model/Classification.kt` |
| Vision | `vision/ImagePreprocessor.kt` |
| Classifier | `classifier/SkinClassifier.kt` |
| State | `state/DiagnosisViewModel.kt` |
| UI | `ui/DownloadScreen.kt`, `ui/CaptureScreen.kt`, `ui/CameraCaptureScreen.kt`, `ui/ResultsScreen.kt`, `ui/Components.kt` |

## Build & run

```bash
cd apps/Skin-Image-Classification/Android
# point local.properties at your Android SDK (or open in Android Studio):
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties
./gradlew :app:assembleDebug
./gradlew :app:installDebug   # with a device connected
```

- **Toolchain:** AGP 8.6.1, Kotlin 2.2.20, Gradle 8.9, compileSdk 35, **minSdk 31**
  (required by the Melange `runtimes` AAR), Jetpack Compose + Material3.
- **SDK:** `com.zeticai.mlange:mlange:1.8.1` (Maven Central). `useLegacyPackaging = true` is
  required so the native `.so` runtimes are extracted on-device. **1.8.x is mandatory** — the
  model is published as an ExecuTorch FP32 target, which only core ≥ 0.1.1 (mlange 1.8.x) can
  load; 1.6.1 throws `No enum constant ...Target.EXECUTORCH_FP32`.
- The classifier model downloads once on first launch (needs connectivity), then runs fully
  offline. 1.8.x performs an online backend-selection handshake on each cold start.

Set your Melange key before building: run `./adapt_mlange_key.sh` from the repo root (it
replaces the `YOUR_MLANGE_KEY` placeholder in `core/AppConfig.kt`'s `PERSONAL_KEY`), or paste
your token in directly. Get one free at <https://mlange.zetic.ai>.

## Model on the Melange dashboard

| Role | Melange name | Version | Mode |
|------|--------------|---------|------|
| Classifier | `realtonypark/Skin_Cancer-Image_Classification` | 1 | `ModelMode.RUN_AUTO` |

## ⚠️ Validate the classifier before demoing

The PyTorch ViT expects **NCHW `[1,3,224,224]`, RGB, normalized to `[-1,1]`**
(`px/127.5 − 1`). The Melange-converted graph *might* expect a different **layout (NHWC),
channel order (BGR), or normalization** — the #1 source of a "wrong prediction on device"
bug. All three are tunable in `AppConfig.Preprocess`. Sweep them (≤12 combinations) against
known Python `argmax` results until the on-device argmax agrees, and confirm
`SkinClass.ordered` matches the model's `id2label`.

## Notes & safety

- All inference is on-device; nothing is uploaded.
- A **non-dismissible disclaimer** is always shown; the UI avoids the word "diagnosis".
  Guidance copy in `SkinClass` escalates for malignant/pre-cancerous classes.

_Demo only — not a medical device._

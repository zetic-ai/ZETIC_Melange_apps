# Skin Image Classification — On-Device (iOS)

> App / target: **SkinImageClassification** · display name **"Skin Classifier"** ·
> bundle id `ai.zetic.demo.SkinImageClassification`.

A demo for skin-image AI companies showing that **ZETIC Melange runs your own models
fully on-device** — no cloud, no data leaving the phone. A real dermatology **vision
model** classifies a skin photo's actual pixels, entirely offline:

- **Skin vision model** — a ViT image classifier (`Anwarkh1/Skin_Cancer-Image_Classification`,
  7 HAM10000 lesion types) deployed on Melange, run via `ZeticMLangeModel`.

The pitch in one screen: *"swap the model `name` for your own Melange model and the rest
of the app is unchanged — your dermatology model, on-device, no cloud."*

> **Note on MedGemma.** This began as a two-model pipeline (classifier → MedGemma-4b LLM
> for a plain-language explanation). On real devices the 4B LLM's on-device load was the
> bottleneck, so the LLM step was dropped in favor of curated, per-condition guidance
> text. The LLM code (`Services/MedGemmaService.swift`, `Services/Prompts.swift`,
> `Services/LLMOutput.swift`, `Components/StreamingText.swift`) remains in the project,
> unused, so it can be re-enabled once a smaller/quantized MedGemma is deployed. (The
> Melange LLM API is text-only — no image input — so MedGemma would explain the
> classifier's *result*, not see the photo.)

## Architecture

```
Photo (camera / library)
  └─ ImagePreprocessor   UIImage → 224×224 RGB → Float[1,3,224,224] (NCHW, [-1,1])
       └─ SkinClassifier   ZeticMLangeModel.run(inputs:) → 7 logits → softmax
            └─ Classification (top class + confidence + ranked distribution)
                 └─ ResultsView   verdict card + distribution bars + curated guidance
```

The classifier loads on a background queue with download progress (cached after first
run). Inference is ~tens of ms on-device; `MemoryProbe` logs RSS at each stage.

## Project layout

| Area | Files |
|------|-------|
| App / config | `App/SkinImageClassificationApp.swift`, `App/AppConfig.swift` (key, model name, **preprocess config**, Debug), `App/Theme.swift` |
| Models | `Models/SkinClass.swift` (7 classes + severity + curated guidance), `Models/Classification.swift` |
| Vision | `Vision/ImagePreprocessor.swift`, `Vision/PhotoPickerView.swift` |
| Services | `Services/SkinClassifier.swift` (active). `MedGemmaService.swift`, `Prompts.swift`, `LLMOutput.swift` (retained, unused) |
| Pipeline / state | `Pipeline/PipelineState.swift`, `State/DiagnosisViewModel.swift` |
| UI | `Views/*` (Root, Download, Capture, Results), `Components/*` (ConfidenceRing, ClassDistributionBars, DisclaimerBanner; StreamingText retained, unused) |

## Build & run

**Device-only** — the Melange `ZeticMLange.xcframework` (SwiftPM, pinned to **exact
1.6.0**) ships no Simulator slice, so it runs on a physical iPhone only.

```bash
cd apps/Skin-Image-Classification/iOS
xcodegen generate          # produces SkinImageClassification.xcodeproj
open SkinImageClassification.xcodeproj     # select your device + signing team, then Run
```

CLI build + install (used during development):

```bash
xcodebuild -project SkinImageClassification.xcodeproj -scheme SkinImageClassification \
  -destination 'id=<DEVICE_UDID>' -allowProvisioningUpdates \
  DEVELOPMENT_TEAM=<TEAM_ID> build
xcrun devicectl device install app --device <DEVICE_UDID> <built>.app
```

The classifier model downloads once on first launch and is cached.

## Model on the Melange dashboard

Configured in `App/AppConfig.swift`:

| Role | Melange name | Version | Mode |
|------|--------------|---------|------|
| Classifier | `realtonypark/Skin_Cancer-Image_Classification` | 1 | `ModelMode.RUN_AUTO` |

To demo a client's model, deploy theirs and change the `name`/`version` (and, for a
classifier, the labels in `Models/SkinClass.swift` + the input contract below).

## ⚠️ Validate the classifier before demoing (most important step)

The PyTorch ViT expects **NCHW `[1,3,224,224]`, RGB, normalized to `[-1,1]`**
(mean=std=0.5 → `px/127.5 − 1`), confirmed against `../model_conversion/`. The
Melange-converted graph *might* expect a different **layout (NHWC), channel order (BGR),
or normalization** — the #1 source of a "wrong prediction on device" bug. All three are
tunable in `AppConfig.Preprocess`:

```swift
enum Preprocess {
    static let layout: Layout = .nchw          // .nchw | .nhwc
    static let channelOrder: ChannelOrder = .rgb // .rgb | .bgr
    static let normalize: Normalize = .signed1   // .signed1 [-1,1] | .unit [0,1]
}
```

**Procedure:**
1. In Python, run the model on ~5 sample lesion images, record the `argmax`.
2. Run the same images through the app. If the on-device `argmax` matches → done.
3. If not, sweep the three axes above (≤12 combinations) until argmax agrees, then leave
   the winning combo in `AppConfig`.
4. Confirm the **label order**: `SkinClass.allCases` must equal the model's `id2label`
   (verified against `../model_conversion/output/labels.json`).

## Diagnostics

`AppConfig.Debug.selfTestClassifierOnLaunch` runs the pipeline on a generated image at
launch and writes the outcome to `Documents/selftest.log` (pull with `devicectl device
copy from … --domain-type appDataContainer`). **Set it to `false` for shipping demos.**

## Notes & safety

- All inference is on-device; nothing is uploaded.
- A **non-dismissible disclaimer** is always shown; the UI avoids the word "diagnosis".
  Guidance copy in `SkinClass` escalates for malignant/pre-cancerous classes.
- `MemoryProbe` logs resident memory at each stage.

_Demo only — not a medical device._

# Aiberry — Berry Check-in (on-device demo)

A SwiftUI demo built to pitch **Aiberry**: a Botberry-style **guided multimodal
mood check-in that runs 100% on-device** via ZETIC Melange. You talk to an emoting
"Berry" avatar; your **facial expression** (Melange ViT FER) and **voice**
(Melange wav2vec2 SER) are read locally and fused into an explainable, *non-diagnostic*
"Screening Insights" readout. Camera/audio never leave the phone — the HIPAA pitch.

Sibling to `../iOS/VoiceVitals`; reuses its Melange plumbing and design language
(warm cream + sage, serif editorial) plus one raspberry "berry" accent.

## What runs on-device
- **Face emotion** — Elena Ryumina's ResNet50/AffectNet model, **already hosted on
  Melange** as `ElenaRyumina/FaceEmotionRecognition` (v1). Faces are found/cropped by
  Apple **Vision** (no extra model). Input contract (done in `FacePixelTensor`):
  `1×3×224×224` NCHW, **BGR**, raw 0–255 with per-channel mean subtraction
  `[91.4953, 103.8827, 131.0912]` (no /255, no std); output is 7 raw logits in
  `[Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger]`, remapped to the
  app's canonical labels.
- **Voice emotion** — wav2vec2 SER on Melange (reused from VoiceVitals).
- **Transcript** — Apple `SFSpeechRecognizer` with `requiresOnDeviceRecognition`.
- **Fusion** — a transparent valence/arousal rule (`Session/FusionEngine.swift`),
  not a trained head: Mood, Energy, Rate of Speech → composite well-being.

## Models
Both models are already hosted on Melange — **no upload step**. `AppConfig.Model.face`
= `ElenaRyumina/FaceEmotionRecognition`, `AppConfig.Model.voice` = the wav2vec2 SER.
Swapping in a client's own model is a one-line `AppConfig` change.

## Build & run
ZeticMLange ships **device-only** slices (no simulator). Compile-check without a device:

```bash
xcodebuild build -scheme Aiberry -project iOS-Aiberry/Aiberry.xcodeproj \
  -destination 'generic/platform=iOS' -configuration Debug CODE_SIGNING_ALLOWED=NO
```

Full run requires a physical iPhone (camera + mic + NPU). On first launch both models
download/compile, then cache (works offline afterwards — try Airplane Mode).

## On-device checklist
- Both models preload; "Start check-in" enables when ready.
- Front-camera self-view (PiP) shows an upright face; the Berry avatar reacts ~3 Hz.
  If the cropped face is ever upside-down, flip `FaceDetector.verticalFlip`.
- Complete the questions → Insights shows the gauge, sub-dimensions, emotion blend,
  and an on-device transcript, with the non-diagnostic banner.

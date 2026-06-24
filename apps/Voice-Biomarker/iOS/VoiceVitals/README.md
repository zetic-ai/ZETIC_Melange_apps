# VoiceVitals — On-Device Voice Biomarker Demo

A SwiftUI iPhone app that runs voice-AI models **fully on-device** through
[ZETIC Melange](https://docs.zetic.ai), accelerated on the Apple Neural Engine.
Built to show voice-biomarker companies (Aiberry, Ellipsis Health, RAIsonance,
Silvia) that they can deploy **their own** models on-device the same way.

**The pitch:** the microphone audio never leaves the phone (HIPAA-friendly), it
runs offline (Airplane Mode), inference is milliseconds on the NPU, and there is
no per-inference cloud-GPU cost. Swapping in a client's model is a one-line change.

## Two tabs, one pipeline

| Tab | Melange model | License | What it shows |
|-----|---------------|---------|---------------|
| **Emotion** | `r-f/wav2vec-english-speech-emotion-recognition` (wav2vec2-large-xlsr) | Apache-2.0 | 7 emotions (angry/disgust/fear/happy/neutral/sad/surprise). A proxy for a mental-health vocal biomarker. |
| **Respiratory** | `google/Sound Classification(YAMNET)` | Apache-2.0 | Cough / breath / wheeze detection (the RAIsonance use case). |

> The Emotion model decides from **prosody** (pitch, energy, rate, voice quality), not
> word meaning. It was trained on acted English (RAVDESS/TESS/SAVEE); cross-corpus
> accuracy is ~50–55% (validated on CREMA-D), with confusions between acoustically
> adjacent emotions. It is a demo, not a clinical measure.

## Run it

1. Open `VoiceVitals.xcodeproj` in Xcode 16+.
2. The Swift package `ZeticMLangeiOS` is pinned to **exact 1.6.0** and resolves on open.
3. Select your team under Signing & Capabilities (bundle id `ai.zetic.VoiceVitals`).
4. Build & run on a **physical iPhone** (the Melange runtime is device-only — there
   is no simulator slice; the NPU is the point).
5. Both models are downloaded/compiled **at launch** (progress shown per tab), then cached.
   The Emotion model is wav2vec2-large (~1.27 GB fp32 before Melange quantization), so the
   first download/compile takes a while; it is cached afterwards.

Note: the app target links **Accelerate.framework** (`OTHER_LDFLAGS`), required by the
ZeticMLange runtime.

Recording: **Emotion** is tap-to-start / tap-to-stop (any length; short clips are tiled,
long clips centre-cropped to the 3 s window); **Respiratory** auto-stops after 3 s.

Microphone permission is requested on first record. The key (`AppConfig.personalKey`)
is the ZETIC dev key; replace with your own from https://mlange.zetic.ai → Settings.

## How a model gets here

PyTorch/HuggingFace → **PT2** (`torch.export`, PyTorch ≥ 2.9) + a fixed-shape
`.npy` sample input → upload at https://mlange.zetic.ai → reference it by `name`.
The 7-class Emotion model is exported with `../SpeechEmotion/prepare/prepare_model_7class.py`
(artifacts: `model_7class.pt2`, `input_0_7class.npy`, `labels_7class.json`). That script
reconstructs the repo's custom head, **bakes in input normalization** (`do_normalize=True`),
and validates against real CREMA-D clips before export. YAMNet is already hosted in Melange.
(`prepare_model.py` still exports the simpler 4-class base model if you want a snappier swap.)

To deploy your own: upload, then change one line —
```swift
ZeticMLangeModel(personalKey:, name: "your-org/your-model", version: 1, modelMode: .RUN_AUTO)
```

## Architecture

```
AudioRecorder (16 kHz mono) ─┬─ EmotionModel → wav2vec2-large (1,48000) → 7 logits → softmax
                             └─ YamnetModel  → YAMNet (48000) → [frames,521] → events
```

- `Core/` — config, one-shot recorder, ZeticMLange helpers, shared model store (launch preload).
- `Emotion/` — model wrapper + polished result UI (per-emotion emoji/colour, hero card).
- `UI/` — shared privacy banner, latency badge, record button, bars, theme.

Built and verified against ZeticMLange 1.6.0 (`xcodebuild` for `generic/platform=iOS`).

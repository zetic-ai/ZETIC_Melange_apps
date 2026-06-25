# Aiberry — Berry Check-in (Android, on-device demo)

A Jetpack Compose port of [`../iOS-Aiberry`](../iOS-Aiberry): a Botberry-style **guided
multimodal mood check-in that runs 100% on-device** via ZETIC Melange. You talk to an
emoting "Berry" avatar; your **facial expression** (Melange FER) and **voice**
(Melange wav2vec2 SER) are read locally and fused into an explainable, *non-diagnostic*
"Screening Insights" readout. Camera/audio never leave the phone — the HIPAA pitch.

Platform pair of `iOS-Aiberry`; reuses the Melange/audio plumbing and design language
(warm cream + sage, serif editorial) of the sibling `../Android` (VoiceVitals) app, plus
one raspberry "berry" accent.

## What runs on-device
- **Face emotion** — Elena Ryumina's ResNet/AffectNet model, hosted on Melange as
  `ElenaRyumina/FaceEmotionRecognition` (v1). Faces are found/cropped by **ML Kit**
  face detection (the Android parallel to Apple Vision). Input contract (`FacePixelTensor`):
  `1×3×224×224` NCHW, **BGR**, raw 0–255 with per-channel mean subtraction
  `[91.4953, 103.8827, 131.0912]` (no /255, no std); output is 7 logits in
  `[Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger]`, remapped to the app's
  canonical labels.
- **Voice emotion** — `realtonypark/Wav2Vec2-Base_Emotion-Recognition` (v2) on Melange.
- **Transcript** — full on-device Whisper-base: log-mel (`WhisperMel`) → `OpenAI/whisper-base-encoder`
  → greedy decode against `OpenAI/whisper-base-decoder` → BPE detokenize (`WhisperTokenizer`).
- **Fusion** — a transparent valence/arousal rule (`session/FusionEngine.kt`), not a trained
  head: Mood, Energy, Rate of Speech → composite well-being. Byte-for-byte the iOS formula
  (locked by `FusionEngineTest`).

## Melange SDK note
Pinned to **`com.zeticai.mlange:mlange:1.8.1`** (not the `1.6.1` in the model snippets): the
v2 voice model is an ExecuTorch-FP32 target that only loads on core ≥ 0.1.1 (mlange 1.8.x).
1.8.x does an online backend-selection handshake on each process cold start, so the **first**
launch needs network; model bytes are cached afterward. All model init/run is funnelled onto one
shared single-thread executor (`MelangeRuntime`) — the SDK forbids concurrent init.

## Build & run
ZeticMLange ships **arm64 device-only** native libraries (no x86 emulator inference).
Compile-check without a device:

```bash
cd Android-Aiberry && ./gradlew assembleDebug      # build the APK
./gradlew :app:testDebugUnitTest                    # FusionEngine numeric-parity tests
```

Full run requires a **physical arm64 Android phone** (camera + mic + NPU), minSdk 31. On first
launch the models download/compile, then cache.

## On-device checklist
- Grant camera + mic; both emotion models preload; "Start check-in" enables when ready.
- Front-camera self-view (PiP) shows an upright face; the Berry avatar reacts ~3 Hz.
  If the cropped face is ever rotated, adjust the rotation handling in `FaceEmotionModel.rotateUpright`.
- Complete the questions → Insights shows the gauge, sub-dimensions, emotion blend, and an
  on-device transcript, with the non-diagnostic banner.
- **Whisper decoder I/O is the main unknown to confirm on-device**: `WhisperTranscriber` logs the
  encoder/decoder tensor shapes at runtime; reconcile `decodeGreedy` against ZETIC's published
  Whisper example if transcripts come back empty. The app degrades gracefully (empty transcript +
  on-device note) if decode can't complete — the face+voice fusion is unaffected.

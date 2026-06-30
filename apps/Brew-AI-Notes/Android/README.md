# Brew — Android (Kotlin)

Native Android port of the iOS **Brew** AI meeting-notes app, built with Jetpack
Compose + Room and the **ZeticMLange** on-device SDK. Record a meeting →
transcribe it on-device with **Whisper** → turn the transcript into a clean
Markdown note with on-device **Gemma** → ask questions about it in a grounded
chat. Everything runs locally; nothing leaves the device after the one-time model
download.

## Models (Melange)

| Role | Model | Class | Mode |
|------|-------|-------|------|
| Note enhance + chat | `changgeun/gemma-4-E2B-it` (v1) | `ZeticMLangeLLMModel` | `RUN_SPEED` |
| STT encoder | `vaibhav-zetic/whisper_small_encoder` (v3) | `ZeticMLangeModel` | `RUN_AUTO` |
| STT decoder | `vaibhav-zetic/whisper_small_decoder` (v1) | `ZeticMLangeModel` | `RUN_AUTO` |

The Whisper pipeline is implemented around the raw encoder/decoder tensors:
`PCM 16 kHz mono → log-mel [1,80,3000] → encoder → hidden [1,1500,768] →
greedy decoder loop (forced English) → GPT-2 byte-level BPE detokenize`. The
log-mel (`mel_filters_80.bin`) and vocab (`vocab.json`) assets are bundled.

## Build & run

```bash
# From this directory
./gradlew :app:assembleDebug         # builds app-debug.apk
./gradlew :app:installDebug          # install on a connected device
./gradlew :app:testDebugUnitTest     # pure-logic unit tests (sanitize/prompts/windowing)
```

> [!IMPORTANT]
> The ZeticMLange native engine is **arm64-only** — run on a **physical arm64
> device**. On an x86_64 emulator the models will not load (the UI stays
> navigable, but transcription/AI no-op). First launch downloads the models, so
> the device needs network once.

The Melange personal key is set in `BrewConfig.kt`.

## Build configuration (the load-bearing bits)

- `com.zeticai.mlange:mlange:1.6.1` from Maven Central; AGP 8.7.3 / Kotlin 2.1.0 /
  Gradle 8.9; `compileSdk 35`, `minSdk 31` (the `runtimes` floor), JVM 17.
- `packaging { jniLibs { useLegacyPackaging = true; pickFirsts += libc++_shared.so ×4 ABIs } }`
  — required so the bundled `.so` libs extract and load (else `UnsatisfiedLinkError`).
- `-keep class com.zeticai.** { *; }` in `proguard-rules.pro` (defensive; R8 is off for debug).
- Permissions: `RECORD_AUDIO`, `INTERNET`.

## Architecture

```
com/brew/
  data/      Room — NoteEntity, ChatMessageEntity, DAOs, NotesRepository
  audio/     WavAudioRecorder (AudioRecord PCM_FLOAT → WAV) + WavIo
  asr/       WhisperService (enc/dec + forced-EN greedy), LogMel, WhisperDetokenizer,
             TranscriptionWorker (background + crash recovery)
  llm/       LLMService (Gemma streaming), Prompts, LLMOutput.sanitize
  engine/    ModelCoordinator — sequential model ownership (never both loaded → no OOM)
  vm/ ui/    ViewModels + Compose screens (list, recording, note detail, chat, settings)
```

`ModelCoordinator` enforces that the Gemma LLM and the Whisper encoder/decoder are
**never co-resident** (they would OOM mid-range devices): transcription loads
Whisper then frees it; enhance/chat then loads Gemma.

## Scope

This port targets the **core flow**: record → transcribe → AI note → per-note
chat, plus the home list, note detail (Note/Transcript tabs), and minimal
settings. **English only.**

Deferred (present in iOS, not yet here): global "Ask Anything", Markdown/Plain/PDF
export & share, and settings storage breakdown beyond counts.

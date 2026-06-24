# OfflineTranslator — Offline Translation Android App (ZETIC.ai Melange demo)

A Jetpack Compose Android translator styled after **DeepL's** app, running a translation model
**fully on-device / offline** via ZETIC.ai's **Melange** (`com.zeticai.mlange:mlange`) SDK. This is
a Kotlin port of the SwiftUI [`iOS/`](../iOS) app, behavior- and pixel-faithful.

The deployed model is **`palm/tencent_HY-MT`** (Tencent Hunyuan-MT), version 1, run in `RUN_AUTO`
mode — configured in [`config/ZeticConfig.kt`](app/src/main/java/ai/zetic/demo/offlinetranslator/config/ZeticConfig.kt).
Text-to-speech (speaker buttons) uses the on-device `android.speech.tts.TextToSpeech`, so audio
readout also works fully offline.

## What it does

- DeepL-style dark UI: idle → editing → result flow, persistent **source ⇄ target** language bar
  with swap, large searchable language list, default **Detect language → English (US)**.
- Streams tokens from the on-device model into the result view as they generate (live caret).
- Live **Online/Offline** badge + "powered by Zetic" — the only branding.
- Offline-friendly extras: **Copy**, **Share** (Android Sharesheet), **Speak** (TextToSpeech).
- **Voice input** (offline speech-to-text) via **ML Kit GenAI Speech Recognition** (BASIC mode) —
  `service/VoiceInputController.kt`. One-time on-device model download, then fully offline.
- **Image input / OCR** (offline) via **ML Kit Text Recognition** (bundled Latin + CJK + Devanagari
  models), from camera or gallery — `ocr/ImageTextRecognizer.kt`. The idle screen's mic/camera
  buttons capture → recognize → translate.
- Permissions: `RECORD_AUDIO` (voice), `CAMERA` (image capture); gallery uses the Android Photo
  Picker (no permission). A `FileProvider` supplies the camera-capture Uri.

## Architecture (mirrors the iOS layers)

| Layer | File |
|------|------|
| App entry | `MainActivity.kt` |
| Theme (programmatic dark palette) | `theme/Theme.kt` |
| Languages, config | `model/Language.kt`, `config/ZeticConfig.kt` |
| Engine abstraction + prompt builder + cleanup + factory | `translation/Translator.kt` |
| Real SDK engine | `translation/ZeticTranslator.kt` |
| Mock engine (emulator / no-arm64 fallback) | `translation/MockTranslator.kt` |
| Streaming state (single-thread engine + token loop) | `viewmodel/TranslationViewModel.kt` |
| Live network status / on-device TTS | `service/NetworkMonitor.kt`, `service/SpeechController.kt` |
| Screen flow + components | `ui/TranslatorScreen.kt`, `ui/IdleView.kt`, `ui/EditingView.kt`, `ui/ResultView.kt`, `ui/LanguagePickerView.kt`, `ui/components/*` |

> **Threading (important):** the ZeticMLange native model is blocking and not thread-safe. All
> engine calls (`load`/construct, `run`, `waitForNextToken`, `cleanUp`, `deinit`) run on **one
> dedicated single-thread `Executor`** in `TranslationViewModel` — never a coroutine `IO`/`Default`
> pool, which would crash the native init. A generation-id `AtomicInteger` supersedes an in-flight
> translation when a newer one starts (mirrors the iOS `genLock`).

The engine is behind the `Translator` interface. `TranslatorFactory.create()` picks the real
`ZeticTranslator` on arm64 hardware, else `MockTranslator` (runtime equivalent of the iOS
`#if canImport(ZeticMLange)` two-target split), so the full UI is demoable on x86_64 emulators.

The prompt uses Tencent **Hunyuan-MT**'s official template
([`TranslationPrompt`](app/src/main/java/ai/zetic/demo/offlinetranslator/translation/Translator.kt)):
`Translate the following segment into {Target}, without additional explanation.\n\n{text}`, with a
Chinese-instruction variant (`把下面的文本翻译成…`) when the target is Chinese. Only the target is
specified — the model infers the source — which is why "Detect language" needs no source in the
prompt.

## Requirements

- **`minSdk 31`** (Android 12). This is a hard floor: `com.zeticai.mlange:runtimes` declares
  `minSdk 31`. `compileSdk`/`targetSdk` 35.
- Real on-device inference requires a physical **arm64-v8a** device. The SDK ships native arm64
  libraries (QNN / TFLite / ONNX Runtime / llama.cpp backends); x86_64 emulators fall back to the
  mock engine.
- The `packaging { jniLibs { useLegacyPackaging = true } }` block in `app/build.gradle.kts` is
  **required** so the SDK's `.so` files are extracted and loadable at runtime.

## Build & run

```bash
# From this Android/ directory. Uses the Gradle wrapper (Gradle 8.9, AGP 8.7.3, Kotlin 2.1.0).
./gradlew assembleDebug        # build the debug APK
./gradlew installDebug         # install on a connected device/emulator
```

`local.properties` must point at your Android SDK (`sdk.dir=…`). The dependency
`com.zeticai.mlange:mlange:1.6.1` resolves from **Maven Central** (no custom repo needed).

### Real offline translation (physical device)

1. Open in Android Studio (or `./gradlew installDebug`) and run on a physical **arm64** phone.
2. First launch downloads `palm/tencent_HY-MT` v1 once (progress shows in the loading overlay).
3. Then enable **Airplane Mode** and translate — it runs fully on-device.

> The `personalKey` in `config/ZeticConfig.kt` is the `YOUR_MLANGE_KEY` placeholder — run
> `./adapt_mlange_key.sh` from the repo root (or paste your Melange Personal Access Token) before building.

## Verification status

- ✅ `./gradlew assembleDebug` builds successfully (Gradle 8.9 / AGP 8.7.3 / Kotlin 2.1.0,
  Compose BOM 2024.12.01); `com.zeticai.mlange:mlange:1.6.1` resolves from Maven Central.
- ✅ The debug APK packages the arm64-v8a native engine libraries (`libzetic_mlange_*`,
  `libllama`, `libggml-*`, `libQnn*`).
- ⤷ Live on-device inference must be run on a physical arm64 device (the SDK has no working
  x86_64 inference slice; emulators show the mock engine).

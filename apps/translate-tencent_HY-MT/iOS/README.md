# OfflineTranslator — Offline Translation iOS App (ZETIC.ai Melange demo)

A SwiftUI iOS translator styled after **DeepL's** app, but running a translation model
**fully on-device / offline** via ZETIC.ai's **Melange** (`ZeticMLange`) SDK. Built to show
DeepL that they could ship an offline mode by deploying their own model on-device with a tiny
integration surface.

The deployed model is **`vaibhav-zetic/tencent_HY-MT`** (Tencent Hunyuan-MT), version 1, run in
`.RUN_AUTO` mode — configured in `Config/ZeticConfig.swift` / `Translation/ZeticTranslator.swift`.
The app just consumes whatever model `ZeticConfig` points at (here Tencent Hunyuan-MT). Text-to-speech
(speaker buttons) uses on-device `AVSpeechSynthesizer`, so audio readout also works fully offline.

## What it does

- DeepL-style dark UI: idle → editing → result flow, persistent **source ⇄ target** language
  bar with swap, large language list, default **Detect language → English**.
- Streams tokens from the on-device model into the result view as they generate.
- Subtle **Offline** badge + "powered by Zetic" — the only branding.
- Offline-friendly extras: **Copy**, **Share**, **Speak** (AVSpeechSynthesizer).
- **Voice input** (offline speech-to-text) via Apple's **Speech** framework
  (`SFSpeechRecognizer` with `requiresOnDeviceRecognition`) — `Services/SpeechInputController.swift`.
- **Image input / OCR** (offline) via Apple's **Vision** framework (`VNRecognizeTextRequest`),
  from camera or photo library — `Services/VisionTextRecognizer.swift`, `Views/ImagePicker.swift`.
  The idle screen's mic/camera buttons capture → recognize → translate.
- Permissions (set as `INFOPLIST_KEY_*` build settings): microphone, speech recognition, camera,
  photo library.

## Architecture

| Layer | File |
|------|------|
| App entry | `OfflineTranslator/OfflineTranslatorApp.swift` |
| Screen flow (idle/editing/result + header + language bar) | `Views/TranslatorScreen.swift`, `Views/IdleView.swift`, `Views/EditingView.swift`, `Views/ResultView.swift` |
| Language picker / components / theme | `Views/LanguagePickerView.swift`, `Views/Components/*`, `Theme/DeepLTheme.swift` |
| Streaming view model (serial-queue load + token loop) | `ViewModels/TranslationViewModel.swift` |
| Engine abstraction + prompt builder | `Translation/Translator.swift` |
| Real SDK engine (device only) | `Translation/ZeticTranslator.swift` |
| Mock engine (simulator/preview) | `Translation/MockTranslator.swift` |
| Languages, config | `Models/Language.swift`, `Config/ZeticConfig.swift` |

> **Threading (important):** the ZeticMLange native model is blocking and not thread-safe.
> All engine calls (`load`, `run`, `waitForNextToken`, `cleanUp`) run on **one dedicated serial
> `DispatchQueue`** in `TranslationViewModel` — never Swift's cooperative pool (`Task.detached`),
> which crashes the native `init` with `EXC_BAD_ACCESS`. This mirrors every shipping ZeticMLange app.

The translation engine is behind the `Translator` protocol. The concrete type is chosen with
`#if canImport(ZeticMLange)`; `ZeticTranslator.swift` is wrapped in the same guard, so it
compiles to nothing where the SDK isn't linked. The prompt uses Tencent **Hunyuan-MT**'s
official template (`Translation/Translator.swift`): `Translate the following segment into
{Target}, without additional explanation.\n\n{text}`, with a Chinese-instruction variant
(`把下面的文本翻译成…`) when the target is Chinese. Only the target is specified — the model
infers the source — which is why "Detect language" needs no source in the prompt. Using the
generic "Translate … to X" phrasing instead made Hunyuan unreliable (it would sometimes echo
the source language rather than translate).

### Two targets (because the SDK is device-only)

`ZeticMLange` 1.6.0 ships **only an `ios-arm64` device slice** (no simulator slice, min iOS 16.6),
so the project has two app targets sharing the same sources:

- **`OfflineTranslator`** — depends on the `ZeticMLange` SPM package (exact `1.6.0`) + `Accelerate`.
  The real offline build; runs on a physical iPhone.
- **`OfflineTranslatorPreview`** — no SDK; uses `MockTranslator` so the **UI** can be built, run,
  and screenshotted on the iOS Simulator.

> Note: the SDK's static ggml CPU backend references BLAS/vDSP, so the app target links
> `Accelerate.framework` (declared in `project.yml`).

## Build & run

The project is generated with [XcodeGen](https://github.com/yonsm/XcodeGen) from `project.yml`.

```bash
xcodegen generate          # produces OfflineTranslator.xcodeproj
open OfflineTranslator.xcodeproj
```

### UI on the simulator (mock engine)

```bash
xcodebuild -scheme OfflineTranslatorPreview \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' build
```
Run it; the mock streams a canned translation so every screen is exercisable. For a quick
demo of a specific state, pass `TG_SEED=editing` or `TG_SEED=result`:
```bash
SIMCTL_CHILD_TG_SEED=result xcrun simctl launch --terminate-running-process \
  "iPhone 17 Pro" ai.zetic.demo.offlinetranslator.preview
```

### Real offline translation (physical device)

1. Open the project, select the **`OfflineTranslator`** scheme.
2. Set a Signing Team (Signing & Capabilities) and pick your iPhone (arm64, **iOS 16.6+**).
3. Run. First launch downloads the model once (progress shows in the header badge).
4. Then put the device in **Airplane Mode** and translate — it runs fully on-device.

> The `personalKey` in `Config/ZeticConfig.swift` is the `YOUR_MLANGE_KEY` placeholder — run
> `./adapt_mlange_key.sh` from the repo root (or paste your Melange Personal Access Token) before building.

## Verification status

- ✅ `OfflineTranslatorPreview` builds for the simulator; idle/editing/result screens match the
  DeepL reference screenshots and the result streams token-by-token.
- ✅ `OfflineTranslator` (model `vaibhav-zetic/tencent_HY-MT` v1, `.RUN_AUTO`) compiles
  **and links** against the real `ZeticMLange` 1.6.0 `ios-arm64` slice
  (`xcodebuild -destination 'generic/platform=iOS' build CODE_SIGNING_ALLOWED=NO`).
- ⤷ Live on-device inference must be run on a physical iPhone (the SDK has no simulator slice).

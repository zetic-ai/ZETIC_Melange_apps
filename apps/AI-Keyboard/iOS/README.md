# CherryPad — On-Device AI Keyboard (iOS)

CherryPad is an on-device AI keyboard, a [MangoPad](https://apps.apple.com/us/app/mangopad-ai-keyboard/id6747285343)-style
clone that runs entirely on the iPhone via [ZETIC.ai Melange](https://docs.zetic.ai/). It offers four
AI actions — **Rewrite** (with tones), **Reply** (Agreeable/Disagreeable), **Translate**, and **Grammar**
— powered by a small Qwen instruct model. No text ever leaves the device.

## Architecture

Because an iOS custom-keyboard extension is jetsam-limited to ~60–120 MB, **no usable LLM fits inside
the keyboard**. So:

- **Container app (`CherryPad`)** — the primary surface. Hosts the full MangoPad-style compose UI and
  the warm `ZeticMLangeLLMModel`. All inference runs here.
- **Keyboard extension (`CherryPadKeyboard`)** — a standard QWERTY plus an AI action bar. It runs no
  model: an action captures the nearby/selected text, writes it to a shared **App Group**, and
  deep-links (`cherrypad://`) into the container app, which generates and writes the result back. The
  keyboard then offers **Insert result**, and the result is also on the pasteboard.

```
Host app → CherryPadKeyboard (capture text → App Group → cherrypad://) → CherryPad app (infer) → result → Insert/paste
```

## Model

One model handles all four tasks via prompting (see `CherryPad/Services/Prompts.swift`):

- Default **`Qwen/Qwen3-0.6B`** — Qwen3, needs `/no_think`, ~0.4 GB.
- Optional **`Steve/LFM2.5_350M`** ("LFM2.5 350M" in Settings) — a Liquid LFM2.5 model that is **non-reasoning**
  (no thinking), so no `/no_think`; loaded with `modelMode: .RUN_ACCURACY` per its dashboard recipe. Often gives
  cleaner rewrites/replies than 0.6B. ~0.3 GB.
- Avoid: `Steve/Qwen3.5-2B` always emits a long visible "Thinking Process:" that can't be suppressed;
  `Qwen2.5-1.5B` / `Qwen3-1.7B` 404 for this key.

**Inference notes (learned the hard way, verified on device):**
- The SDK's `run(_:)` applies the model's **chat template internally** — pass plain instruction text, never
  raw ChatML or your own `User:`/`Assistant:` labels (either corrupts output into repetitive garbage).
- Qwen3 needs **`/no_think`** in the prompt or it loops forever in `<think>` and returns nothing usable.
- Use **minimal `ZeticMLangeLLMModel` init** (key/name/version only). Passing `modelMode: .RUN_SPEED` + a
  custom `LLMInitOption(nCtx:)` made Qwen3-0.6B degenerate even with `/no_think`.
- 0.6B is small: Reply/Grammar are reliable; Rewrite uses a "keep all info / don't shorten" + trailing-prime
  prompt to avoid collapsing; Translate is coherent but modest quality — use the 4B tier for better results.

Latency work: model loaded once at launch and kept **warm** (only `cleanUp()` between turns), capped output
tokens, streamed tokens into the result card.

## On-device model test

`CherryPadTests` (a `bundle.unit-test` target) loads the real model and runs the actual prompts, asserting
coherent (non-degenerate) output — the way to verify device-only inference without UI tapping:

```bash
xcodebuild test -project CherryPad.xcodeproj -scheme CherryPad \
  -destination 'id=<your-device-udid>' -allowProvisioningUpdates
```

The Melange key is the placeholder `"YOUR_MLANGE_KEY"` in `CherryPad/Services/ZeticConfig.swift` — run
the repo-root `./adapt_mlange_key.sh` to inject the real key (keeps it out of git).

## Build

Requires [`xcodegen`](https://github.com/yonaskolb/XcodeGen).

```bash
cd apps/AI-Keyboard/iOS
xcodegen generate

# Device compile-check (app + embedded keyboard, real SDK):
xcodebuild build -project CherryPad.xcodeproj -scheme CherryPad \
  -destination 'generic/platform=iOS' -configuration Debug CODE_SIGNING_ALLOWED=NO

# Simulator UI iteration (no SDK; StubLLMEngine streams canned text):
xcodebuild build -project CherryPad.xcodeproj -scheme CherryPadPreview \
  -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO
```

`generic/platform=iOS` is required for the `CherryPad` scheme — the ZeticMLange 1.6.0 SPM package ships
an arm64 **device-only** slice, so a Simulator destination can't link it. The `CherryPadPreview` target
is package-free and selects `StubLLMEngine` via `#if targetEnvironment(simulator)` so the UI runs on the
Simulator.

## Using the keyboard in other apps (e.g. Notes)

**One-time setup:**
1. Settings ▸ General ▸ Keyboard ▸ Keyboards ▸ **Add New Keyboard** ▸ CherryPad.
2. Settings ▸ General ▸ Keyboard ▸ Keyboards ▸ **CherryPad** ▸ turn on **Allow Full Access**.
   Without Full Access the AI actions can't open the app or share your text — this is the #1 reason
   "it doesn't work". (The plain QWERTY still types without it.)

**Each use (in Notes):**
1. Type or paste text; **select** the part you want to transform.
2. Tap 🌐 to switch to the CherryPad keyboard (cherry-red action bar: Rewrite / Reply / Translate / Grammar).
3. Tap an action → CherryPad opens, generates, and **copies the result to the clipboard**.
4. Switch back to Notes and **paste** to replace your text — or reopen the CherryPad keyboard and tap
   **Insert result** to drop it at the cursor.

If tapping an action does NOT auto-open CherryPad (iOS sometimes blocks the launch), just open CherryPad
manually from the Home Screen — it picks up your pending text on launch and processes it. iOS provides no
way to auto-return you to Notes, so the paste / Insert-result step is always manual.

## On-device test (the AI + keyboard handoff are device-only)

1. Set the signing team (`SF5GDD3C2G`) and register the App Group `group.ai.zetic.demo.cherrypad` on the
   Apple Developer portal for both targets.
2. `./adapt_mlange_key.sh` from the repo root to inject the Melange key.
3. Run **CherryPad** on a device; let the model download (progress UI).
4. Settings ▸ General ▸ Keyboard ▸ Keyboards ▸ Add New Keyboard ▸ **CherryPad**, then tap CherryPad and
   enable **Allow Full Access** (required for the App Group + launching the app).
5. In Notes/Messages: type or select text, switch to CherryPad with 🌐, tap an action → CherryPad opens
   pre-filled and streams the result → **Apply** (copies) or reopen the keyboard and tap **Insert result**.

### Verify each feature (in-app)
- **Rewrite**: cycle the tone chips; meaning and language are preserved.
- **Reply**: toggle Agreeable/Disagreeable; the stance flips.
- **Translate**: pick a target language; output is that language only.
- **Grammar**: fixes errors; already-correct text is returned unchanged.
- **Latency**: first token should appear in well under a second on the warm `.fast` model.

## Layout

```
CherryPad/            container app (SwiftUI)
  App/                AppModel, Theme
  Services/           LLM engine (+ Simulator stub), ZeticConfig, Prompts, LLMOutput, KeyboardBridge
  Models/             Language
  Views/              ComposeScreen, ActionBar, ChipRow, LanguagePickerView, ResultCard,
                      ModelDownloadView, OnboardingView, SettingsView, RootView
CherryPadKeyboard/    keyboard extension (no SDK): KeyboardViewController, KeyboardView, KeyboardActionBar
Shared/               compiled into both: AppGroup, HandoffPayload, DeepLink, KeyboardTask, Tone, Stance
```

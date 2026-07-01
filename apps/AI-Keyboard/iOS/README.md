# CherryPad — On-Device AI Keyboard (iOS)

CherryPad is an on-device AI keyboard, a [MangoPad](https://apps.apple.com/us/app/mangopad-ai-keyboard/id6747285343)-style
clone that runs entirely on the iPhone via [ZETIC.ai Melange](https://docs.zetic.ai/). It offers four
AI actions — **Rewrite** (with tones), **Reply** (Agreeable/Disagreeable), **Translate**, and **Grammar**
— powered by the small non-reasoning **LFM2.5-350M** model. No text ever leaves the device.

iOS is SwiftUI; the Android port lives at [`apps/AI-Keyboard/Android`](../Android) and is behaviorally
identical (same prompts, tones, stances, language list, output sanitizer, cherry-red design).

## Architecture

The model runs **in-process inside the keyboard extension** — tap an action and the result is prepared
right there, no app-switch:

- **Keyboard extension (`CherryPadKeyboard`)** — a full QWERTY plus a cherry-red AI action bar. An action
  captures the selected/surrounding text, runs LFM2.5-350M **in the extension** (`KeyboardLLM`), and offers
  **Insert result** (replaces the selection / inserts at the cursor). This is the primary surface.
- **Container app (`CherryPad`)** — a MangoPad-style compose screen (Rewrite/Reply/Translate/Grammar with
  tone chips, a language picker, a streaming result card), plus Settings and Onboarding.

Both share the App Group `group.ai.zetic.demo.cherrypad` (the app writes the chosen translate target; the
keyboard reads it). **Full Access is required** so the keyboard can download the model on first use.

### Fitting an LLM in a keyboard extension

iOS keyboard extensions get only ~48–60 MB of **dirty** working memory. The trick: the model *weights* are
memory-mapped (clean/evictable, mostly uncounted); what counts is the **KV cache**, which grows per
generated token. CherryPad keeps the working set under the limit with the smallest model (**LFM2.5-350M**),
a small context (`nCtx = 512`), a hard **64-token** output cap, and truncated input — reliable for the short
text a keyboard is actually used for. (See `CherryPadKeyboard/KeyboardLLM.swift`.)

## Model

One model handles all four tasks via prompting (see `Prompts.swift`):

- **`Steve/LFM2.5_350M`** (version 1), loaded with `RUN_ACCURACY`. Small (~0.3 GB), non-reasoning (no
  `<think>`), fully on-device after a one-time download.

**Inference notes:**
- The SDK's `run(_:)` applies the model's **chat template internally** — pass plain instruction text, never
  raw ChatML or your own `User:`/`Assistant:` labels (either corrupts output into repetitive garbage).
- Output is streamed and sanitized (`LLMOutput.swift`) to strip stray `<think>`, ChatML control tokens,
  role/task labels, and wrapping quotes.

The Melange key is the placeholder `"YOUR_MLANGE_KEY"` in `CherryPad/Services/ZeticConfig.swift` — run the
repo-root `./adapt_mlange_key.sh` to inject your token.

## Build

Requires [`xcodegen`](https://github.com/yonaskolb/XcodeGen).

```bash
cd apps/AI-Keyboard/iOS
xcodegen generate

# Device (app + embedded keyboard, real SDK):
xcodebuild build -project CherryPad.xcodeproj -scheme CherryPad \
  -destination 'generic/platform=iOS' -configuration Debug CODE_SIGNING_ALLOWED=NO

# Simulator UI (no SDK; stub engine):
xcodebuild build -project CherryPad.xcodeproj -scheme CherryPadPreview \
  -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO
```

`generic/platform=iOS` is required for the `CherryPad` scheme — the ZeticMLange 1.6.0 SPM package ships an
arm64 **device-only** slice. The `CherryPadPreview` target is package-free and uses a `StubLLMEngine` so the
SwiftUI UI runs on the Simulator.

> [!TIP]
> **Iterate on the keyboard UI in the Simulator**: launch the preview app with the `-kbDesign` argument to
> render the `KeyboardDesignHarness` (every keyboard state at the real fixed height) for screenshots.

## Enable & use the keyboard

**One-time setup:**
1. Install and open **CherryPad** (onboarding shows the steps; the model downloads on first keyboard use).
2. Settings ▸ General ▸ Keyboard ▸ Keyboards ▸ **Add New Keyboard** ▸ CherryPad, then tap CherryPad and turn
   on **Allow Full Access** (required for the model download).

**Each use (in any app, e.g. Notes):**
1. Type or paste text; optionally **select** the part you want to transform.
2. Tap 🌐 to switch to CherryPad.
3. Tap an action — **Rewrite / Reply / Translate / Grammar**. The model runs on the keyboard and the AI panel
   takes over the keyboard area with a result card.
4. Tap **Insert result** to drop it in place, or ✕ to dismiss.

The translate target language is chosen in the app (Translate ▸ language chip) and reused by the keyboard.

## Layout

```
project.yml                  xcodegen: CherryPad (app) + CherryPadKeyboard (ext) + CherryPadPreview + CherryPadTests
CherryPad/                   container app: App/, Services/ (LLM engine, ZeticConfig, Prompts, LLMOutput), Models/, Views/
CherryPadKeyboard/           keyboard extension: KeyboardViewController, KeyboardView, KeyboardActionBar, KeyboardState,
                             KeyboardLLM (in-process inference)
CherryPadPreview/            Simulator-only @main + KeyboardDesignHarness (-kbDesign)
Shared/                      App Group, KeyboardTask, Tone, Stance
```

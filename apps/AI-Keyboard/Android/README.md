# CherryPad — On-Device AI Keyboard (Android)

CherryPad is an on-device AI keyboard, a [MangoPad](https://apps.apple.com/us/app/mangopad-ai-keyboard/id6747285343)-style
clone that runs entirely on the phone via [ZETIC.ai Melange](https://docs.zetic.ai/). It offers four
AI actions — **Rewrite** (with tones), **Reply** (Agreeable/Disagreeable), **Translate**, and **Grammar**
— powered by the small non-reasoning **LFM2.5-350M** model. No text ever leaves the device.

This is the Android port of `apps/AI-Keyboard/iOS` (CherryPad). It is behaviorally identical: the same
prompts, tones, stances, 30-language list, output sanitizer, and cherry-red design.

## Architecture

Unlike iOS — where a keyboard extension is jetsam-limited to ~60 MB and can't hold a usable LLM — an
Android `InputMethodService` (IME) runs inside the app's own process with normal app-level memory. So
**the keyboard runs LFM2.5-350M in-process directly**, with no app round-trip:

- **Keyboard (`CherryImeService`)** — a full QWERTY plus a cherry-red AI action bar. An action captures
  the selected/surrounding text, runs the model in-process, and offers **Insert result** (which replaces
  the selection, or inserts at the cursor). This is the primary surface.
- **Container app (`MainActivity`)** — the MangoPad-style compose screen (Rewrite/Reply/Translate/Grammar
  with tone chips, a language picker, a streaming result card), plus Settings and Onboarding.

Both surfaces share one process, so they share a single warm model (`LLMService`, an `object`) and a
single `SharedPreferences` (`Prefs`) — the app writes the chosen translate target, the keyboard reads it.
No App Group, deep-link handoff, or "Full Access" gate is needed (those are iOS constraints).

## Model

One model handles all four tasks via prompting (see `llm/Prompts.kt`):

- **`Steve/LFM2.5_350M`** (version 1), loaded with `LLMModelMode.RUN_AUTO`. Small (~0.3 GB),
  non-reasoning (no `<think>`), fully on-device after a one-time download.

**Inference notes (ported from iOS, verified there on device):**
- The SDK's `run(_)` applies the model's **chat template internally** — pass plain instruction text, never
  raw ChatML or your own `User:`/`Assistant:` labels (either corrupts output into repetitive garbage).
- Use minimal init (key/name/version/mode). Output is streamed token-by-token and sanitized
  (`llm/LLMOutput.kt`) to strip stray `<think>`, ChatML control tokens, role/task labels, and wrapping quotes.
- Per-task output budgets: grammar 256, rewrite 256, reply 200, translate 320.

The Melange key lives in `llm/ZeticConfig.kt` (`PERSONAL_KEY`).

## Build

Requires the Android SDK (platform 34/35, build-tools 34+). JDK 17+.

```bash
cd apps/AI-Keyboard/Android
# local.properties should point at your SDK, e.g.:
#   sdk.dir=$HOME/Library/Android/sdk
./gradlew :app:assembleDebug
```

Notes:
- `mlange:1.6.1` resolves from Maven Central. `packaging { jniLibs { useLegacyPackaging = true } }` is
  **required** so the SDK's `.so` files are extracted (otherwise `UnsatisfiedLinkError`).
- The ZeticMLange native engine is **arm64-only**: the model loads on real devices / arm64 emulators.
  On x86_64 emulators the keyboard still types, but AI actions no-op (the model won't load).

## Enable & use the keyboard

**One-time setup:**
1. Install and open **CherryPad** (it warms/downloads the model; onboarding shows the steps).
2. Settings ▸ System ▸ Languages & input ▸ On-screen keyboard ▸ Manage keyboards ▸ turn on **CherryPad**.
   (Onboarding's **Open Settings** button jumps you here.)

**Each use (in any app):**
1. Type or paste text; optionally **select** the part you want to transform.
2. Tap the keyboard-switch icon (or 🌐) and pick **CherryPad**.
3. Tap an action — **Rewrite / Reply / Translate / Grammar**. The model runs on the keyboard and shows a
   result card.
4. Tap **Insert result** to drop it in place (replacing the selection, or inserting at the cursor).

The translate target language is chosen in the app (Translate ▸ language chip) and reused by the keyboard.

## Layout

```
app/src/main/java/ai/zetic/demo/cherrypad/
  MainActivity.kt              container app entry
  AppModel.kt                  compose-screen state machine (ViewModel)
  llm/    ZeticConfig, Prompts, LLMOutput, LLMService (shared warm model)
  model/  KeyboardTask, Tone, Stance, Language
  data/   Prefs (SharedPreferences: targetLang, hasSeenOnboarding)
  ui/     theme/, RootScreen, ComposeScreen, ActionBar, ChipRow, ResultCard,
          ModelStatusBanner, LanguagePickerScreen, SettingsScreen, OnboardingScreen
  keyboard/ KB (design tokens), KeyboardState, KeyboardScreen (QWERTY + AI bar),
            CherryImeService (InputMethodService + Compose host + in-process inference)
```

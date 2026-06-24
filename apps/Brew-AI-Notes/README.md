# Brew — AI Meeting Notes

<div align="center">

**On-Device AI Meeting Notes powered by Gemma (`gemma-4-E2B-it`)**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](iOS/)

</div>

<div align="center">

| | | |
|:---:|:---:|:---:|
| <img src="../../res/screenshots/ainotes.gif" width="180" alt="Demo 1"> | <img src="../../res/screenshots/brew-main.png" width="240" alt="Brew Main"> | <img src="../../res/screenshots/brew-note.png" width="240" alt="Brew Note"> |

</div>

> [!TIP]
> **View on Melange Dashboard**: [changgeun/gemma-4-E2B-it](https://mlange.zetic.ai/p/changgeun/gemma-4-E2B-it)

Brew records your meetings, transcribes them on-device, and uses an on-device **Gemma** model (via **Melange**) to turn the raw transcript into a clean, structured note — then lets you **Ask Anything** about it in a chat. Audio, transcription, and AI generation all run locally; nothing leaves the device.

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
   This replaces the `YOUR_MLANGE_KEY` placeholder in `iOS/Brew/Services/LLMService.swift` with your personal access token.
3. **Run the App**:
   - **iOS**: Open `iOS/` in Xcode and run on a physical device

> [!NOTE]
> The Gemma model only runs on **physical devices** (it uses the NPU via Melange). On the **iOS Simulator** the app falls back to a built-in `StubLLMEngine`, so the UI stays navigable without real inference.

## 📚 Resources

- **Melange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/changgeun/gemma-4-E2B-it)
- **Base Model**: Gemma (Google), instruction-tuned `E2B` variant
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Model**: `gemma-4-E2B-it`
- **Task**: Conversational LLM — note enhancement, title generation, and Q&A chat
- **Melange Project**: [changgeun/gemma-4-E2B-it](https://mlange.zetic.ai/p/changgeun/gemma-4-E2B-it)
- **Architecture**: Gemma (Decoder-only Transformer)
- **Key Features**:
  - Fully on-device inference via Melange
  - Real-time token streaming
  - Character-budgeted prompts sized against a 4096-token context window

This application showcases the **Gemma** model using **Melange**. Combined with Apple's on-device speech recognition, Brew delivers a complete private meeting-notes workflow — record, transcribe, summarize, and chat — without a network round trip.

## 📁 Directory Structure

```
Brew-AI-Notes/
└── iOS/                            # iOS implementation with SwiftUI, SwiftData & Melange SDK
    ├── project.yml                 # XcodeGen project definition
    ├── Vendor/
    │   └── ZeticMLange.xcframework # Vendored Melange SDK (device + empty Simulator slice)
    ├── Brew.xcodeproj/
    └── Brew/
        ├── BrewApp.swift           # App entry point (SwiftData model container)
        ├── Models/                 # Note & ChatMessage (SwiftData models)
        ├── Views/                  # SwiftUI screens (notes list, detail, recording, chat)
        ├── ViewModels/             # Recording, NoteDetail, Chat & AskAnything view models
        ├── Services/
        │   ├── LLMService.swift        # Single owner of the local model; engine selection
        │   ├── ZeticLLMEngine.swift    # Zetic MLange (Gemma) integration — device builds
        │   ├── StubLLMEngine.swift     # Simulator fallback engine
        │   ├── SpeechTranscriber.swift # On-device Apple Speech transcription
        │   ├── AudioRecorder.swift     # AVFoundation meeting capture
        │   ├── Prompts.swift           # Enhance / title / chat prompt builders
        │   └── NoteExporter.swift      # Export notes
        └── Theme/                  # Shared visual theme
```

## 🔧 Technical Details

### Model Architecture

- **Base Model**: Gemma (`gemma-4-E2B-it`)
- **Input Format**: Raw text prompt (transcript + instruction, or chat history)
- **Output Format**: Streaming tokens
- **Context Length Management**: Character-budgeted prompts (`truncateMiddle` keeps the head and tail of long transcripts), sized conservatively (~3 chars/token) against the 4096-token window

### Inference Process

1. **Single Owner**: `LLMService` funnels all engine access through one serial executor/queue so the single generation context is never used concurrently. (iOS additionally swaps in a `StubLLMEngine` on the Simulator.)
2. **Download Handling**: The app surfaces real model download progress (`Downloading`) and an indeterminate `Preparing` state during weight loading/compilation.
3. **Transcription**: The app runs Apple Speech over the recorded file.
4. **Prompt Building**: `Prompts` builds enhance / title / chat prompts, trimming overlong transcripts to fit the context budget. The prompt templates and budgets are carefully tuned.
5. **Token Streaming**: Tokens stream from `model.waitForNextToken()` back to the UI (an `AsyncThrowingStream` on iOS).

### Key Implementation Details

- **On-Device Privacy**: Recording, transcription, and AI generation all happen locally — no audio or text leaves the device.
- **Vendored SDK**: The upstream Melange package ships a device-only slice, which breaks Simulator builds. This app vendors `ZeticMLange.xcframework` with an empty Simulator slice and links it as a static framework (`embed: false`); the Simulator build references no symbols from it and uses `StubLLMEngine` instead.
- **Resilient Capture**: A model failure never blocks recording — audio still captures and transcribes, and the AI note is generated later.

## 💡 Features

- ✅ **On-Device Transcription**: English and Korean speech recognition, fully offline (Apple Speech on iOS).
- ✅ **AI Note Enhancement**: Gemma turns raw transcripts into clean, structured meeting notes with a generated title.
- ✅ **Ask Anything**: Chat with the model about any note, with real-time token streaming.
- ✅ **Private by Design**: Audio, transcript, and AI output never leave the device.
- ✅ **Hardware Accelerated**: Fully optimized via Melange for the mobile NPU.
- ✅ **Native UI**: SwiftUI + SwiftData on iOS.

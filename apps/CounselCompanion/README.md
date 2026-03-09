# Counsel Companion

<div align="center">

**On-Device AI Mental Wellness Companion powered by Kanana 1.5 (2.1B)**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](android/)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](ios/)

</div>

> [!TIP]
> **View on Melange Dashboard**: [Steve/kanana_1_5__2_1b_instruct](https://mlange.zetic.ai/p/Steve/kanana_1_5__2_1b_instruct)

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the App**:
   - **Android**: Open `android/` in Android Studio
   - **iOS**: Open `ios/` in Xcode

## 📚 Resources

- **Melange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/Steve/kanana_1_5__2_1b_instruct)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Model**: Kanana 1.5 (2.1B Instruct)
- **Task**: Conversational LLM (Mental Wellness Counseling)
- **Melange Project**: [Steve/kanana_1_5__2_1b_instruct](https://mlange.zetic.ai/p/Steve/kanana_1_5__2_1b_instruct)
- **Architecture**: Decoder-only Transformer
- **Key Features**:
  - Fully on-device inference via Melange
  - Real-time token streaming
  - Multi-session conversation management
  - Customizable system prompt for counseling persona

This application showcases a **mental wellness companion chatbot** using **Melange**. The app provides a warm, supportive chat interface running an LLM completely on-device for maximum privacy.

## 📁 Directory Structure

```
CounselCompanion/
├── android/       # Android implementation with Jetpack Compose & Melange SDK
│   └── app/
│       └── src/main/
│           ├── java/com/yeonseok/melangecounsel/
│           │   ├── MainActivity.kt              # Main UI Entry Point
│           │   ├── llm/ZeticChatEngine.kt       # Zetic MLange Model Integration
│           │   ├── ui/viewmodel/CounselViewModel.kt  # State Management
│           │   ├── ui/screens/                   # Jetpack Compose Screens
│           │   │   ├── ChatScreen.kt             # Chat Interface
│           │   │   ├── SessionsScreen.kt         # Session List
│           │   │   ├── SettingsScreen.kt         # Theme & Prompt Settings
│           │   │   └── DiagnosticsScreen.kt      # Model Diagnostics
│           │   └── data/                         # Room Database & Repositories
│           └── AndroidManifest.xml
└── ios/           # iOS implementation with SwiftUI & Melange SDK
    └── CounselCompanion/
        ├── App/App.swift                         # App Entry Point
        ├── ViewModel/ChatViewModel.swift         # State Management
        ├── LLM/ZeticChatEngine.swift             # Zetic MLange Model Integration
        ├── Views/
        │   ├── ChatScreen.swift                  # Chat Interface
        │   ├── SessionsScreen.swift              # Session List
        │   ├── SettingsScreen.swift               # Theme & Prompt Settings
        │   └── DiagnosticsScreen.swift           # Model Diagnostics
        └── Persistence/                          # Local Session & Settings Storage
```

## 🔧 Technical Details

### Model Architecture

- **Base Model**: Kanana 1.5 2.1B Instruct
- **Input Format**: Raw Text Prompt (with system prompt and conversation history)
- **Output Format**: Streaming Tokens
- **Context Management**: History-aware prompt building with system prompt injection

### Inference Process

1. **Initialization**: Model is loaded asynchronously in the background on app launch.
2. **Download Handling**: App visualizes model download progress with a linear progress bar.
3. **Prompt Building**: `ModelCompatibilityLayer` constructs prompts with system prompt, conversation history, and current user input.
4. **Token Streaming**: The UI streams `model.waitForNextToken()` results in real-time with Markdown rendering.
5. **Memory Management**: `model.cleanUp()` is called before/after generation to manage KV cache resources.

### Key Implementation Details

- **MVVM + Repository Pattern**: Clean separation between UI, business logic, and data layers.
- **Multi-Session Support**: Users can create and switch between multiple conversation sessions.
- **Customizable Persona**: System prompt is editable to adjust the counselor's behavior.
- **Diagnostics Dashboard**: Real-time metrics including token count, generation duration, and stop reason.
- **Warm Color Palette**: Sage and peach tones designed for a calming counseling experience.

## 💡 Features

- ✅ **Real-time Streaming**: See the LLM response generated token-by-token with Markdown support.
- ✅ **Dynamic Progress Indicator**: Visual feedback during initial model download.
- ✅ **Multi-Session Management**: Create, switch, and manage multiple conversation sessions.
- ✅ **Customizable System Prompt**: Tailor the counselor's personality and approach.
- ✅ **Theme Support**: System, Light, and Dark mode options.
- ✅ **Model Diagnostics**: View generation metrics, token counts, and performance data.
- ✅ **Complete Privacy**: All inference runs on-device with no data leaving the phone.
- ✅ **Cross-Platform**: Native UI for both Android (Compose) and iOS (SwiftUI).

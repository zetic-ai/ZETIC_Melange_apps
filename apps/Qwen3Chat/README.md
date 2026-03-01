# Qwen3-4B Chat App

<div align="center">

<div align="center">

| **iOS** |
|:---:|
| <img src="../../res/screenshots/qwen_4b_ios.gif" width="200" alt="iOS Screenshot"> |

</div>


**On-Device LLM Chatbot leveraging Qwen3-4B**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](Android/)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](iOS/)

</div>

> [!TIP]
> **View on Melange Dashboard**: [Qwen/Qwen3-4B](https://mlange.zetic.ai/p/Qwen/Qwen3-4B)

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the App**:
   - **Android**: Open `Android/` in Android Studio
   - **iOS**: Open `iOS/` in Xcode

## 📚 Resources

- **Melange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/Qwen/Qwen3-4B)
- **Base Model**: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on Hugging Face
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Model**: Qwen3-4B
- **Task**: Conversational LLM (Chat)
- **Melange Project**: [Qwen/Qwen3-4B](https://mlange.zetic.ai/p/Qwen/Qwen3-4B)
- **Base Model**: [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) on Hugging Face
- **Architecture**: Qwen (Decoder-only Transformer)
- **Key Features**:
  - Fully on-device inference via Melange
  - Real-time token streaming
  - Sliding context window for optimized memory management

This application showcases the **Qwen3-4B** model using **Melange**. The app provides a fully functional native chat interface, running an advanced large language model completely on-device.

## 📁 Directory Structure

```
Qwen3Chat/
├── Android/       # Android implementation with Jetpack Compose & Melange SDK
│   └── app/
│       └── src/main/
│           ├── java/com/zeticai/qwen3chat/
│           │   ├── MainActivity.kt        # Main UI Entry Point
│           │   ├── llm/LLMService.kt      # Zetic MLange Model Integration
│           │   └── ui/                    # Jetpack Compose Screens
│           └── AndroidManifest.xml
└── iOS/          # iOS implementation with SwiftUI & Melange SDK
    └── Qwen3Chat.xcodeproj/
    └── Qwen3Chat/
        ├── Qwen3Chat_iOSApp.swift         # App Entry Point
        └── View/
            ├── ChatView.swift             # Main Chat Interface
            ├── LLMService.swift           # Zetic MLange Model Integration
            └── ChatSessionManager.swift   # History & Prompt Builder
```

## 🔧 Technical Details

### Model Architecture

- **Base Model**: Qwen3-4B
- **Input Format**: Raw Text Prompt
- **Output Format**: Streaming Tokens
- **Context Length Management**: Managed dynamic sliding window (`maxCharacters = 3000`)

### Inference Process

1. **Initialization**: The model is eagerly loaded in the background via `Task.detached` (iOS) or `viewModelScope.launch` (Android).
2. **Download Handling**: App visualizes the model disk download progress state natively.
3. **Prompt Building**: `ChatSessionManager` maintains a character-limited trailing context window for conversation history, appending `\nAssistant: ` to prime generation.
4. **Token Streaming**: The UI loops `model.waitForNextToken()` to stream text back to the screen seamlessly.
5. **Memory Management**: The framework ensures KV Cache is purged properly utilizing `model.cleanUp()` post-generation loops and on cancellation.

### Key Implementation Details

- **Responsive Token Streaming**: Tokens yield immediately to the Main Thread/Actor.
- **Sliding History Window**: Drops the oldest messages to avoid KV-token overflow exceptions.
- **Native Gestural UI**: Implements standard iOS (`scrollDismissesKeyboard`) and Android interactive bounds.

## 💡 Features

- ✅ **Real-time Streaming**: See the LLM response generated token-by-token.
- ✅ **Dynamic Progress Indicator**: Feedback during the initial ~4GB model disk fetch.
- ✅ **Contextual Conversation**: Application remembers and sends recent context in sequential turns.
- ✅ **Hardware Accelerated**: Fully optimized via Melange for Mobile Neural Engines.
- ✅ **Cross-Platform**: Modern Native UI available for both Android (Compose) and iOS (SwiftUI).

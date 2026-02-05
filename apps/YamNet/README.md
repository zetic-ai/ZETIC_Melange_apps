# YamNet

<div align="center">

**Audio Event Classification and Sound Recognition**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](Android/)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](iOS/)

</div>

> [!TIP]
> **View on Melange Dashboard**: [google/Sound Classification(YAMNET)](https://mlange.zetic.ai/p/google/Sound%20Classification(YAMNET)?from=use-cases) - Contains generated source code & benchmark reports.

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

- **Melange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/google/Sound%20Classification(YAMNET)?from=use-cases)
- **Melange Model Library**: [Model Library](https://mlange.zetic.ai/model-library) 
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Model**: Qualcomm YamNet
- **Task**: Audio Classification / Event Detection
- **Melange Project**: [google/Sound Classification(YAMNET)](https://mlange.zetic.ai/p/google/Sound%20Classification(YAMNET)?from=use-cases)
- **Key Features**:
  - Classification of environmental sounds and audio events
  - Real-time audio processing
  - NPU-accelerated inference via Melange

This application showcases the **Qualcomm YamNet** model using **Melange**. YamNet provides audio event classification and sound recognition capabilities, optimized for mobile devices with NPU acceleration.

## 📁 Directory Structure

```
YamNet/
├── prepare/      # Model & input preparation scripts
├── Android/      # Android implementation with Melange SDK
└── iOS/          # iOS implementation with Melange SDK
```

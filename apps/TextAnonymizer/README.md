# Text Anonymizer

<div align="center">

**Automatic PII Detection and Masking for Privacy Protection**

[![MLange](https://img.shields.io/badge/Powered%20by-MLange-orange.svg)](https://mlange.zetic.ai)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](Android/)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](iOS/)

</div>

> [!TIP]
> **View on MLange Dashboard**: [Steve/text-anonymizer-v1](https://mlange.zetic.ai/p/Steve/text-anonymizer-v1?from=use-cases) - Contains generated source code & benchmark reports.

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your MLange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the App**:
   - **Android**: Open `Android/` in Android Studio
   - **iOS**: Open `iOS/` in Xcode

## 📚 Resources

- **MLange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/Steve/text-anonymizer-v1?from=use-cases)
- **Use Cases**: [Text Anonymizer on Use Cases Page](https://mlange.zetic.ai/use-cases) → [Direct Link](https://mlange.zetic.ai/p/Steve/text-anonymizer-v1?from=use-cases)
- **Documentation**: [MLange Docs](https://docs.zetic.ai)
- **Discord Community**: [Join our Discord](https://discord.com/invite/gVFX6myuMx)

## Overview

The **Text Anonymizer** application provides automatic detection and masking of Personally Identifiable Information (PII) in text using the **text-anonymizer-v1** model powered by **MLange**. This on-device solution ensures that sensitive information such as names, addresses, phone numbers, locations, and dates are automatically identified and replaced with safe placeholders before sharing or processing.

### Key Features

- ✅ **On-Device Processing**: All anonymization happens locally on your device - no data leaves your device
- ✅ **Comprehensive PII Detection**: Detects 5 prominent types of sensitive information
- ✅ **Real-Time Processing**: Fast inference using NPU-optimized models via Zetic MLange
- ✅ **Cross-Platform**: Available for both Android and iOS
- ✅ **User-Friendly UI**: Simple, intuitive interface for text input and anonymized output
- ✅ **Export Options**: Copy to clipboard or share anonymized text directly

## Supported PII Types

The model can detect and mask the following types of sensitive information:

| Label | Placeholder | Description |
|-------|-------------|-------------|
| **PERSON** | `[Person]` | Person names |
| **LOCATION** | `[Location]` | Location names (cities, countries, etc.) |
| **DATE** | `[Date]` | Dates in various formats |
| **ADDRESS** | `[Address]` | Physical addresses |
| **PHONE_NUMBER** | `[Phone number]` | Phone numbers in various formats |
| **O** | (unchanged) | Non-sensitive text |

## Model Information

- **Model Name**: `Steve/text-anonymizer-v1`
- **Model Type**: Token Classification (Named Entity Recognition)
- **Architecture**: RoBERTa-based transformer
- **Input Format**: UTF-8 byte sequences (max 128 bytes)
- **Output Format**: Logits tensor with shape `[1, 128, 11]` (11 classes)
- **Optimization**: NPU-optimized via Zetic MLange for on-device inference

## Project Structure

```
TextAnonymizer/
├── Android/                    # Android application
│   ├── app/
│   │   ├── src/main/
│   │   │   ├── java/com/zeticai/textanonymizer/
│   │   │   │   ├── MainActivity.kt          # Main UI activity
│   │   │   │   ├── AnonymizerViewModel.kt  # ViewModel for model operations
│   │   │   │   └── Constants.kt            # Configuration (API token, model name)
│   │   │   ├── res/                         # Resources (layouts, strings, themes)
│   │   │   └── AndroidManifest.xml
│   │   └── build.gradle.kts                # App-level dependencies
│   ├── build.gradle.kts                    # Project-level config
│   └── README.md                           # Android-specific setup guide
│
├── iOS/                        # iOS application
│   ├── ZeticMLangeTextAnonymizer-iOS/
│   │   ├── TextAnonymizerApp.swift         # App entry point
│   │   ├── ContentView.swift               # Main UI (SwiftUI)
│   │   └── AnonymizerService.swift         # Model handling and inference
│   └── ZeticMLangeTextAnonymizer-iOS.xcodeproj/
│
├── prepare/                     # Model preparation scripts
│   └── extract_tanaos_trace.py # Script to trace model to TorchScript
│
└── README.md                    # This file
```

### Prerequisites

- **MLange Personal Key**: Get your free key from the [MLange Dashboard](https://mlange.zetic.ai)
- **Android**: Android Studio with API 24+
- **iOS**: Xcode 14.0+ with iOS 13.0+

## How It Works

### Model Inference Pipeline

1. **Text Input**: User enters text containing potentially sensitive information
2. **Preprocessing**: 
   - Text is converted to UTF-8 byte sequence
   - Padded or truncated to fixed length (128 bytes)
   - Attention mask is created to indicate valid bytes
3. **Model Inference**:
   - Input tensors are created (`input_ids`, `attention_mask`)
   - Model runs inference using Zetic MLange SDK
   - Output logits tensor contains class probabilities for each byte position
4. **Post-processing**:
   - Logits are converted to predicted labels using argmax
   - Consecutive tokens with the same label form entity spans
   - Spans are mapped to byte positions in the original text
   - Sensitive spans are replaced with appropriate placeholders
5. **Output**: Anonymized text is displayed to the user

### Example

**Input**:
```
My name is Sarah Connor and I live in Los Angeles. 
You can reach me at sarah.connor@example.com or call 555-123-4567.
```

**Output**:
```
My name is [Person] and I live in [Location]. 
You can reach me at [Email] or call [Phone number].
```

## Technical Details

### Model Architecture

- **Base Model**: RoBERTa (Robustly Optimized BERT Pretraining Approach)
- **Task**: Token Classification (Named Entity Recognition)
- **Classes**: 11 classes (O + 10 PII types)
- **Input**: Byte-level encoding (UTF-8)
- **Max Sequence Length**: 128 bytes

### Inference Details

- **Input Tensors**:
  - `input_ids`: `[1, 128]` - Byte values (uint8 or int32)
  - `attention_mask`: `[1, 128]` - Binary mask (1 for valid, 0 for padding)
  
- **Output Tensor**:
  - `logits`: `[1, 128, 11]` - Class probabilities for each byte position
  - Each position has 11 logit values (one per class)

- **Post-processing**:
  - Argmax over classes for each position
  - Group consecutive positions with same label (excluding "O")
  - Map byte positions back to string indices
  - Replace spans with placeholders

### Performance

- **Inference Time**: ~10-50ms per text (device-dependent)
- **Model Size**: ~125MB (optimized for on-device)
- **Memory Usage**: ~200-300MB during inference
- **Accuracy**: High precision and recall on common PII formats

## Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load with authentication error
- **Solution**: Verify your personal access token is correct and valid
- Check that you have internet connection (required for initial download)
- Ensure the model name matches: `jathin-zetic/tanaos-text-anonymizer`

**Problem**: Model loading times out
- **Solution**: Check your internet connection
- The model is downloaded once and cached locally
- First launch may take longer depending on network speed

### Inference Issues

**Problem**: No PII detected in text that clearly contains sensitive information
- **Solution**: 
  - The model works at byte level - ensure text encoding is correct
  - Some formats may not be recognized if they deviate significantly from training data
  - Check console logs for model output details

**Problem**: Incorrect spans or false positives
- **Solution**: 
  - The model uses a threshold-based approach - some edge cases may occur
  - Review the detected spans in console logs
  - Consider post-processing rules for your specific use case

### Build Issues

**Android**:
- Ensure Android SDK is properly configured
- Sync Gradle files: **File** → **Sync Project with Gradle Files**
- Clean and rebuild: **Build** → **Clean Project**, then **Build** → **Rebuild Project**

**iOS**:
- Ensure Xcode Command Line Tools are installed
- Clean build folder: **Product** → **Clean Build Folder** (⇧⌘K)
- Reset package caches if using Swift Package Manager

## Advanced Usage

### Custom Placeholders

You can customize the placeholders used for each PII type by modifying the `placeholderByLabel` dictionary:

**Android** (`AnonymizerViewModel.kt`):
```kotlin
private val placeholderByLabel = mapOf(
    "EMAIL" to "[Email]",
    "PHONE_NUMBER" to "[Phone number]",
    // ... customize as needed
)
```

**iOS** (`AnonymizerService.swift`):
```swift
private let placeholderByLabel: [String: String] = [
    "EMAIL": "[Email]",
    "PHONE_NUMBER": "[Phone number]",
    // ... customize as needed
]
```

### Adjusting Detection Sensitivity

The model uses argmax over logits to determine labels. For more conservative detection (fewer false positives), you could:

1. Add a confidence threshold before accepting predictions
2. Filter out short spans (e.g., < 3 bytes)
3. Apply domain-specific rules post-detection

### Batch Processing

For processing multiple texts, you can extend the ViewModel to handle batches:

```kotlin
fun anonymizeTexts(texts: List<String>): List<String> {
    return texts.map { anonymizeText(it) }
}
```

## Model Preparation

If you need to prepare the model from scratch:

1. **Install dependencies**:
   ```bash
   pip install torch transformers numpy
   ```

2. **Run the extraction script**:
   ```bash
   cd prepare
   python extract_tanaos_trace.py
   ```

3. **Upload to MLange Dashboard**:
   - The script generates a TorchScript model (`.pt` file)
   - Upload this to the MLange Dashboard for optimization
   - The optimized model will be available via the SDK

## Contributing

Contributions are welcome! If you find issues or have suggestions:

1. Check existing issues on the repository
2. Create a new issue with detailed description
3. For code contributions, submit a pull request

## License

This application uses the **tanaos-text-anonymizer-v1** model. Please refer to the model's license on Hugging Face for usage terms.


## Support

For issues, questions, or feature requests:
- **GitHub Issues**: Open an issue on the repository
- **Discord**: Join our community Discord server
- **Email**: Contact support through the MLange Dashboard

---

<div align="center">

**Built with [Zetic MLange](https://mlange.zetic.ai)** - NPU-Native On-Device AI

</div>

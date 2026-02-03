# Chronos Bolt: Time Series Forecasting

<div align="center">

| **iPhone 15 Pro** | **Galaxy S25** |
|:---:|:---:|
| <img src="../../res/screenshots/chronos-recording-ios.gif" width="200" alt="Chronos iOS"> | <img src="../../res/screenshots/chronos-recording-android.gif" width="200" alt="Chronos Android"> |

</div>

<div align="center">

**Probabilistic Time Series Forecasting with Chronos Bolt**

[![Melange](https://img.shields.io/badge/Powered%20by-Melange-orange.svg)](https://mlange.zetic.ai)
[![iOS](https://img.shields.io/badge/Platform-iOS-blue.svg)](iOS/)
[![Android](https://img.shields.io/badge/Platform-Android-green.svg)](Android/)

</div>

> [!TIP]
> **View on Melange Dashboard**: [Team_ZETIC/Chronos-balt-tiny](https://mlange.zetic.ai/p/Team_ZETIC/Chronos-balt-tiny?from=use-cases) - Contains generated source code & benchmark reports.

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the App**:
   - **iOS**: Open `iOS/` in Xcode
   - **Android**: Open `Android/` in Android Studio
   - Build and run on a device or simulator

## 📚 Resources

- **Melange Dashboard**: [View Model & Reports](https://mlange.zetic.ai/p/Team_ZETIC/Chronos-balt-tiny?from=use-cases)
- **Use Cases**: [Chronos Bolt on Use Cases Page](https://mlange.zetic.ai/use-cases) → [Direct Link](https://mlange.zetic.ai/p/Team_ZETIC/Chronos-balt-tiny?from=use-cases)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Model**: Chronos Bolt Tiny
- **Task**: Time Series Forecasting
- **Melange Project**: [Team_ZETIC/Chronos-balt-tiny](https://mlange.zetic.ai/p/Team_ZETIC/Chronos-balt-tiny?from=use-cases)
- **Key Features**:
  - Probabilistic forecasting (quantiles)
  - Zero-shot performance on unseen time series
  - CSV import and table editor; interactive charts
  - NPU-accelerated inference via Melange

This application showcases the **Chronos Bolt Tiny** model using **Melange**. Chronos Bolt is a time series forecasting model optimized for on-device inference. The app supports CSV import, quantile forecasts, and interactive visualization of prediction intervals.

## 📁 Directory Structure

```
ChronosTimeSeries/
├── prepare/      # Model & input preparation scripts
├── iOS/          # iOS implementation with Melange SDK
└── Android/      # Android implementation with Melange SDK
```

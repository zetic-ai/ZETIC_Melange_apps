# PromptGuard (iOS)

Analyzes and classifies prompt injection and jailbreak attacks into harm categories (S1–S11) using the Zetic Melange model **llama_prompt_guard_2**.

## Requirements

- Xcode 15+ (recommended 15.2+)
- iOS 16.0+ (Swift Charts)

## Build & Run

1. **Open the project**
   - In Xcode: **File → Open** and select the folder that contains `ZeticMLangePromptGuard-iOS.xcodeproj` (the `PromptGuard` folder that also contains this README).

2. **Set your Zetic personal key (required)**
   - The app uses `Config.personalKey` for model download. **No keys are committed.**
   - **Option A:** In Xcode: **Product → Scheme → Edit Scheme…** → **Run** → **Arguments** tab → **Environment Variables** → add `ZETIC_PERSONAL_KEY` = your key.
   - **Option B:** In `ZeticMLangePromptGuard-iOS/Core/PromptGuardModel.swift`, replace `YOUR_PERSONAL_KEY` in `Config` with your key (do not commit).

3. **Resolve Swift Package**
   - Xcode will resolve the Zetic Melange dependency automatically (`https://github.com/zetic-ai/ZeticMLangeiOS.git`).
   - If not: **File → Add Package Dependencies** → add `https://github.com/zetic-ai/ZeticMLangeiOS.git`, **Up to Next Major** minimum `1.1.0`, add **ZeticMLange** to the PromptGuard target.

4. **Code signing (required for device)**
   - Select the **PromptGuard** target → **Signing & Capabilities**.
   - Check **Automatically manage signing** and choose your **Team** (Apple ID / development team).

5. **Select device**
   - **Recommended:** Use a **physical iPhone** (iOS 16+). The Zetic Melange binary may not include an iOS Simulator slice; if the scheme only lists “Any iOS Device” or simulator build fails, run on a real device.
   - Select the **PromptGuard** scheme and your connected device in the toolbar.

6. **Build and run**
   - **Product → Run** (or ⌘R).
   - On first launch the app may download the model; ensure the device has network access.
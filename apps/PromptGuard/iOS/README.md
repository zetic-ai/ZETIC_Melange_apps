# PromptGuard (iOS)

**PromptGuard** is an on-device iOS app that classifies text prompts as **Benign** or **Malicious**. It helps you detect prompt-injection and jailbreak-style inputs (e.g. “Ignore your previous instructions”, “Disregard the system prompt”) before or after they are sent to an LLM, so you can block or log them.

The app uses **Llama Prompt Guard 2** (meta-llama/Llama-Prompt-Guard-2-86M) via **Zetic Melange**: the model runs locally on the device (CoreML). No prompt text is sent to the cloud; only the model download uses your Zetic personal key.

---

## What the app does

- **Classify** – You paste a user prompt (and optionally the assistant’s reply). The app tokenizes it with the same tokenizer as the model, runs the classifier on-device, and shows **Benign** or **Malicious** plus the raw logits. Example prompts are available as quick-fill buttons.
- **History** – Every classification is stored locally. You can see recent runs, scores, and a chart of Malicious vs Benign counts. History can be cleared from Settings.
- **Settings** – Dark theme toggle (controls the app’s appearance), data retention (how long to keep history), and clear history. Privacy copy explains that inference is on-device and no prompts are sent to servers.
- **Diagnostics** – Shown from the ⋯ menu: model info, last latency, last error, and **ModelInputSpec** (prompt template with `{user_input}` / `{agent_output}`, max tokens). Useful for debugging and tuning the prompt format.

## Requirements

- Xcode 15+ (recommended 15.2+)
- iOS 16.0+ (Swift Charts)

## Privacy & security

- No API keys, tokens, or personal data are committed. Model download uses `ZETIC_PERSONAL_KEY` from the environment or a local fallback (see Build & Run).
- Classification runs on-device; prompt text is not sent to any server.
- Set your own Apple Development Team in Xcode (Signing & Capabilities) for device runs.

## How to add tokenizer.json (in Xcode)

The app works without it (using a fallback encoding) but needs `tokenizer.json` for accurate, Python-matching classification. Steps:

### Step 1: Generate tokenizer.json (on your Mac, in Terminal)

From the repo root (or from `apps/PromptGuard/prepare`):

```bash
pip install transformers huggingface_hub
export HF_TOKEN=your_huggingface_token   # only if the model is gated on Hugging Face
python apps/PromptGuard/prepare/export_assets.py
```

This creates `apps/PromptGuard/prepare/tokenizer.json` (and optionally `labels.json`).

### Step 2: Add the file in Xcode

1. Open the project in Xcode: open **ZeticMLangePromptGuard-iOS.xcodeproj** (inside `apps/PromptGuard/iOS`).
2. In the **Project Navigator** (left sidebar), right‑click the **ZeticMLangePromptGuard-iOS** group (the yellow folder with the app name).
3. Choose **Add Files to "ZeticMLangePromptGuard-iOS"…**.
4. In the file picker, go to `apps/PromptGuard/prepare` and select **tokenizer.json**.
5. Leave these checked:
   - **Copy items if needed** (so the file is copied into the project folder).
   - **Add to targets: PromptGuard** (so it’s included in the app target).
6. Click **Add**.

### Step 3: Ensure it’s copied into the app bundle

1. In Xcode, select the **PromptGuard** target (blue app icon in the left sidebar).
2. Open the **Build Phases** tab at the top.
3. Expand **Copy Bundle Resources**.
4. If **tokenizer.json** is not in the list, click the **+** button, choose **tokenizer.json**, and click **Add**.

After a clean build and run, the app will load the tokenizer and the info banner about adding tokenizer.json will go away. Without `tokenizer.json`, the app still runs but uses a fallback encoding and shows a short info message.

## Build & Run

1. **Open the project**
   - In Xcode: **File → Open** and select the folder that contains `ZeticMLangePromptGuard-iOS.xcodeproj` (the `PromptGuard` folder that also contains this README).

2. **Set your Zetic personal key (required)**
   - The app uses `Config.personalKey` for model download. **No keys or secrets are committed.**
   - **Option A:** In Xcode: **Product → Scheme → Edit Scheme…** → **Run** → **Arguments** tab → **Environment Variables** → add `ZETIC_PERSONAL_KEY` = your key.
   - **Option B:** In `ZeticMLangePromptGuard-iOS/Core/PromptGuardModel.swift`, replace the fallback in `Config.personalKey` with your key (do not commit).

3. **Resolve Swift Package**
   - Xcode will resolve the Zetic Melange dependency automatically (`https://github.com/zetic-ai/ZeticMLangeiOS.git`).
   - If not: **File → Add Package Dependencies** → add `https://github.com/zetic-ai/ZeticMLangeiOS.git`, **Up to Next Major** minimum `1.1.0`, add **ZeticMLange** to the PromptGuard target.

4. **Code signing (required for device)**
   - Select the **PromptGuard** target → **Signing & Capabilities**.
   - Check **Automatically manage signing** and choose your **Team** (no development team is committed; use your own Apple ID / team).

5. **Select device**
   - **Recommended:** Use a **physical iPhone** (iOS 16+). The Zetic Melange binary may not include an iOS Simulator slice; if the scheme only lists “Any iOS Device” or simulator build fails, run on a real device.
   - Select the **PromptGuard** scheme and your connected device in the toolbar.

6. **Build and run**
   - **Product → Run** (or ⌘R).
   - On first launch the app may download the model; ensure the device has network access.
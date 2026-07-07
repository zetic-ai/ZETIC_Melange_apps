# Pose Motion

**On-Device Human Pose Tracking and 3D Motion Lifting**

> [!TIP]
> **View on Melange Dashboard**: Detector — [vaibhav-zetic/YOLO26n](https://mlange.zetic.ai/p/vaibhav-zetic/YOLO26n) · 2D pose — [realtonypark/RTMPose-s_pose_motion](https://mlange.zetic.ai/p/realtonypark/RTMPose-s_pose_motion) · 3D lift — [realtonypark/MotionBERT-lite_pose_motion](https://mlange.zetic.ai/p/realtonypark/MotionBERT-lite_pose_motion) — Contains generated source code & benchmark reports.

Pose Motion runs a video-based pose pipeline locally with Melange. A person detector finds the
athlete, RTMPose estimates body keypoints frame by frame, and MotionBERT-Lite lifts the rolling
2D keypoint window into a 3D skeleton for motion analysis. The demo is aimed at sports-form and
swing-review workflows where the camera input and inferred body motion should stay on-device.
iOS is SwiftUI, Android is Jetpack Compose.

## 🚀 Quick Start

Get up and running in minutes:

1. **Get your Melange API Key** (free): [Sign up here](https://mlange.zetic.ai)
2. **Configure API Key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the App**:
   - **Android**: Open `Android/` in Android Studio and run on a physical arm64 device.
   - **iOS**: Open `iOS/` in Xcode and run on a physical iPhone.

> A physical device is required for the Melange runtime and camera/video pipeline. The first
> launch downloads and caches the models; after that, inference runs locally.

## 📚 Resources

- **Melange Dashboard**: [Detector](https://mlange.zetic.ai/p/vaibhav-zetic/YOLO26n) · [2D pose](https://mlange.zetic.ai/p/realtonypark/RTMPose-s_pose_motion) · [3D lift](https://mlange.zetic.ai/p/realtonypark/MotionBERT-lite_pose_motion)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## 📋 Model Details

- **Detector**: `vaibhav-zetic/YOLO26n`
  - NMS-free person/object detection used to locate the moving subject.
- **2D pose**: `realtonypark/RTMPose-s_pose_motion`
  - RTMPose small model for 17-keypoint body pose estimation.
- **3D lift**: `realtonypark/MotionBERT-lite_pose_motion`
  - MotionBERT-Lite pose model over an 81-frame keypoint window.
- **Task**: Video frame -> person crop -> 2D skeleton -> 3D pose sequence
- **Key Features**:
  - Fully on-device inference via Melange
  - Real-time 2D overlays plus 3D skeleton visualization
  - Local video processing with cached model downloads

## 📁 Directory Structure

```
Pose-Motion/
├── Android/      # Android implementation with Jetpack Compose and Melange SDK
└── iOS/          # iOS implementation with SwiftUI and Melange SDK
```

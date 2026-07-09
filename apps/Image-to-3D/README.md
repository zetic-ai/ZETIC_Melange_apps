# Image to 3D

**On-Device Photo Depth and 3D Relief Reconstruction**

> [!TIP]
> **View on Melange Dashboard**: [realtonypark/Depth-Anything-V2-Small](https://mlange.zetic.ai/p/realtonypark/Depth-Anything-V2-Small) - Contains generated source code and benchmark reports.

Image to 3D runs a single-photo depth pipeline locally with Melange. A
Depth-Anything-V2-Small model predicts a relative depth map from the selected
photo, then the app converts that depth into an interactive 3D relief mesh or
point cloud. The workflow is built for quick on-device reconstruction previews
where the input photo and inferred geometry stay on the device.

iOS is SwiftUI with SceneKit rendering. Android is Jetpack Compose with an
OpenGL ES relief viewer.

## Quick Start

1. **Get your Melange API key**: [Sign up here](https://mlange.zetic.ai)
2. **Configure API key**:
   ```bash
   # From repository root
   ./adapt_mlange_key.sh
   ```
3. **Run the app**:
   - **Android**: Open `Android/` in Android Studio and run on a physical arm64 device.
   - **iOS**: Open `iOS/` in Xcode and run on a physical iPhone.

> A physical device is required for the Melange runtime. The first launch
> downloads and caches the model; after that, inference runs locally.

## Resources

- **Melange Dashboard**: [Depth-Anything-V2-Small](https://mlange.zetic.ai/p/realtonypark/Depth-Anything-V2-Small)
- **Documentation**: [Melange Docs](https://docs.zetic.ai)

## Model Details

- **Depth model**: `realtonypark/Depth-Anything-V2-Small`
  - Monocular relative depth estimation for a single RGB photo.
- **Task**: Photo -> depth map -> 3D relief mesh or point cloud
- **Input**: `(1, 3, 518, 518)` float32 RGB in `[0, 1]`
- **Output**: `(1, 518, 518)` float32 relative inverse depth
- **Key features**:
  - Fully on-device inference via Melange
  - Interactive textured mesh and colored point-cloud views
  - Local sample-photo self-test flows on both platforms

## Directory Structure

```
Image-to-3D/
|-- Android/      # Android implementation with Jetpack Compose and Melange SDK
`-- iOS/          # iOS implementation with SwiftUI, SceneKit, and Melange SDK
```

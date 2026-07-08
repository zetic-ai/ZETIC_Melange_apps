# Image to 3D — iOS

SwiftUI demo: photo -> on-device depth (Melange) -> interactive 3D.

## Build

```bash
xcodegen generate
open ImageTo3D.xcodeproj
```

Headless check:

```bash
xcodebuild -project ImageTo3D.xcodeproj -scheme ImageTo3D \
  -destination 'generic/platform=iOS' -derivedDataPath build \
  CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO build
```

Runtime is **device-only** (the ZeticMLange xcframework has no simulator slice).

## Setup

1. Confirm the model in `ImageTo3D/App/AppConfig.swift` is Ready on the
   [Melange Dashboard](https://mlange.zetic.ai).
2. Inject your key: run `./adapt_mlange_key.sh` from the repo root
   (the source ships with the `YOUR_MLANGE_KEY` placeholder).

## Pipeline

- `Model/ImagePreprocessor.swift` - EXIF-fix, center-crop square, 518x518, CHW `[0,1]` floats
- `Model/DepthModel.swift` - `ZeticMLangeModel.run` with a `(1,3,518,518)` float32 tensor
- `Model/DepthMap.swift` - defensive output-tensor parse to relative inverse depth
- `Rendering/DepthTo3D.swift` - normalized disparity to relief depth,
  planar relief projection, 518x518 vertex grid, triangles dropped across depth
  discontinuities, photo UVs + per-vertex colors
- `Rendering/Model3DView.swift` - SceneKit: unlit textured mesh / colored point cloud,
  turntable orbit
- `Rendering/DepthColormap.swift` - turbo-colormapped depth pane
- Tunables (FOV, plane depth, relief depth, mesh stride, edge threshold) live in `App/AppConfig.swift`.

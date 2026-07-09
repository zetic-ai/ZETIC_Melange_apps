# Image to 3D — Android

Kotlin/Compose port of the iOS app: photo -> on-device depth (Melange) ->
interactive 3D relief. Same model, same reconstruction algorithm, same layout
(3D pane up top, photo -> depth analysis as the main display).

## Build & run

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.zetic.demo.imageto3d/.MainActivity
```

Headless self-test (bundled sample photo writes results for `adb pull`):

```bash
adb shell am start -n ai.zetic.demo.imageto3d/.MainActivity --ez selftest true
adb shell cat /sdcard/Android/data/ai.zetic.demo.imageto3d/files/selftest/stats.json
```

A pushed self-test input file overrides the bundled sample. Requires an
arm64 device (the Melange native engine has no x86 slice).

## Notes

- **SDK pin `com.zeticai.mlange:mlange:1.8.1`** — the dashboard's model
  manifest includes target names (e.g. `EXECUTORCH_FP32`) that the 1.6.x
  `Target` enum predates; 1.6.1 fails with `No enum constant` at load.
- `useLegacyPackaging = true` + `libc++_shared.so` pickFirsts are required
  (UnsatisfiedLinkError otherwise). INTERNET permission needed for the model
  download on first launch.
- Pipeline: `ImagePreprocessor` (center-crop 518 square, CHW [0,1]) -> `DepthModel`
  (Tensor float32 `[1,3,518,518]` -> `[1,518,518]` inverse depth) ->
  `DepthColormap` (turbo) + `DepthTo3D` (planar relief, raw-disparity quad
  culling; mirrors `iOS/ImageTo3D/Rendering/DepthTo3D.swift` exactly) ->
  `Relief3DView` (GLES 3.0: textured mesh / colored points, idle sway,
  one-finger orbit, pinch zoom).
- Tunables live in `AppConfig.kt`, kept byte-for-byte in sync with the iOS
  `AppConfig.swift`.

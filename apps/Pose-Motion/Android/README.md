# PoseMotion (Android)

Kotlin/Compose port of the iOS PoseMotion demo — the same three Melange models
(YOLO26n -> RTMPose-s -> MotionBERT-Lite), the same bundled golf clips, the same
benchmark HUD. See `../iOS/README.md` for the pipeline description; constants are
1:1 in `AppConfig.kt`.

Android-specific notes:

- **SDK**: `com.zeticai.mlange:mlange:1.8.1` (older 1.6.1 crashes at load on freshly
  converted models — its Target enum is missing constants). `useLegacyPackaging = true`
  is required for the native libs. All models run `ModelMode.RUN_AUTO` (the Android SDK
  has no explicit-target constructor; the iOS CoreML pin was a Metal-specific fix).
- **Deterministic clip decode**: `MediaExtractor` + `MediaCodec` in synchronous
  ByteBuffer mode (`video/VideoFrameDecoder.kt`) — one frame per pull, so Benchmark
  mode measures true sustainable pipeline FPS. YUV→RGB is hand-rolled
  (stride/cropRect-aware, `video/YuvToRgb.kt`) into a 3-deep Bitmap pool.
- **Threading**: one `HandlerThread` owns model construction (the SDK's native init is
  thread-affine), decode, preprocessing, and all inference; results post to Compose
  via StateFlow.
- Model teardown/retry uses `close()` (core 0.1.1 renamed `deinit()`).

## Run

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n ai.zetic.demo.posemotion/.MainActivity
adb logcat -s run:D mem:D    # [mem] <model> loaded / [run] <model> ok (x ms)
```

`local.properties` (`sdk.dir=...`) is machine-local and not committed.

The sample clips ship in `app/src/main/assets/` (CC BY 4.0, see
`ATTRIBUTION.txt`). To try another clip without rebuilding:
`adb push Clip.mp4 /sdcard/Android/data/ai.zetic.demo.posemotion/files/GolfSwing.mp4`

# PoseMotion (iOS)

Sports pose & motion demo on ZETIC Melange: three on-device models run frame-by-frame
over a bundled golf-swing clip with a live per-device benchmark HUD.

| Stage | Model | Melange name |
|---|---|---|
| Person + ball detection | YOLO26n | `vaibhav-zetic/YOLO26n` (v1) |
| 2D skeleton (17 keypoints) | RTMPose-s | `realtonypark/RTMPose-s_pose_motion` |
| 2D→3D lift (81-frame window) | MotionBERT-Lite | `realtonypark/MotionBERT-lite_pose_motion` |

Per frame: YOLO26n finds the athlete (highest-confidence person) and the ball ->
the person box is cropped 3:4 and fed to RTMPose → SimCC argmax-decoded to a COCO-17
skeleton -> converted to H36M-17, pushed into a sliding 81-frame window -> MotionBERT-Lite
lifts the window to a root-relative 3D skeleton (rotatable side view, drag to orbit).

## Modes

- **Benchmark** (default): frames are decoded as fast as inference completes —
  the FPS readout is the true sustainable pipeline throughput on this device.
- **Realtime**: frames are paced to the clip's timestamps (late frames dropped),
  matching a live-capture scenario.

The HUD shows per-model latency (rolling 30-frame mean), sustained FPS (2 s window),
and current/peak process memory footprint.

## Setup

1. Models must be **Ready** on [mlange.zetic.ai](https://mlange.zetic.ai/)
   and match the names, versions, and personal key in `PoseMotion/App/AppConfig.swift`.
2. Sample clips are bundled in `PoseMotion/Media/`. To try another clip, add
   `GolfSwing.mp4`, `GolfSwing2.mp4`, or `GolfSwing3.mp4` there and rebuild, or
   copy one of those filenames into the app's Documents folder via Finder/the
   Files app; the app checks there too.
3. Open `PoseMotion.xcodeproj`, build & run on a device (iOS 16.6+).
   The ZeticMLange SPM package (exact 1.6.0) resolves automatically.

If the 3D lift model is unavailable the demo degrades gracefully to 2D
(detector + skeleton + ball trail still run).

import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Pixel formats we know how to decode from the [camera] plugin.
/// Android streams YUV420, iOS streams BGRA8888.
enum FrameFormat { yuv420, bgra8888 }

/// A camera frame flattened into plain typed-data so it can be shipped to the
/// pipeline isolate (a [CameraImage] itself is not sendable).
class FrameData {
  const FrameData.yuv420({
    required this.width,
    required this.height,
    required this.yPlane,
    required this.uPlane,
    required this.vPlane,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
    this.rotationDegrees = 0,
  })  : format = FrameFormat.yuv420,
        bgra = null,
        bgraRowStride = 0;

  const FrameData.bgra8888({
    required this.width,
    required this.height,
    required Uint8List this.bgra,
    required this.bgraRowStride,
    this.rotationDegrees = 0,
  })  : format = FrameFormat.bgra8888,
        yPlane = null,
        uPlane = null,
        vPlane = null,
        yRowStride = 0,
        uvRowStride = 0,
        uvPixelStride = 0;

  final FrameFormat format;
  final int width;
  final int height;

  /// Clockwise rotation (0/90/180/270) to apply to the raw buffer so the model
  /// sees an upright scene. On the PyroGuard iOS setup the buffer arrived
  /// ALREADY upright (0) — do not assume landscape; measure on-device (the HUD
  /// shows the real buffer WxH). Android delivers sensor orientation (usually
  /// 90) and must be rotated.
  final int rotationDegrees;

  // YUV420 planes.
  final Uint8List? yPlane;
  final Uint8List? uPlane;
  final Uint8List? vPlane;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;

  // BGRA8888 plane.
  final Uint8List? bgra;
  final int bgraRowStride;

  /// Copies the raw planes out of a [CameraImage]. Must be called inside the
  /// image-stream callback (before the plugin recycles the buffer); the bytes
  /// are then owned by this object and safe to send across isolates.
  factory FrameData.fromCameraImage(CameraImage image,
      {int rotationDegrees = 0}) {
    if (image.format.group == ImageFormatGroup.bgra8888) {
      final plane = image.planes.first;
      return FrameData.bgra8888(
        width: image.width,
        height: image.height,
        bgra: Uint8List.fromList(plane.bytes),
        bgraRowStride: plane.bytesPerRow,
        rotationDegrees: rotationDegrees,
      );
    }

    // Default to YUV420 (Android).
    final y = image.planes[0];
    final u = image.planes[1];
    final v = image.planes[2];
    return FrameData.yuv420(
      width: image.width,
      height: image.height,
      yPlane: Uint8List.fromList(y.bytes),
      uPlane: Uint8List.fromList(u.bytes),
      vPlane: Uint8List.fromList(v.bytes),
      yRowStride: y.bytesPerRow,
      uvRowStride: u.bytesPerRow,
      uvPixelStride: u.bytesPerPixel ?? 1,
      rotationDegrees: rotationDegrees,
    );
  }
}

/// Rotation-aware BGR view over a raw [FrameData] buffer.
///
/// Both PP-OCR models are trained on cv2-convention **BGR** input, so all
/// sampling here returns BGR. The upright frame is never materialized: the
/// detector letterbox and the recognizer quad-warp each sample only the
/// pixels they need, straight from the raw planes, with the rotation folded
/// into the coordinate mapping.
class UprightFrame {
  UprightFrame(this.frame)
      : _rot = ((frame.rotationDegrees % 360) + 360) % 360 {
    final swap = _rot == 90 || _rot == 270;
    width = swap ? frame.height : frame.width;
    height = swap ? frame.width : frame.height;
  }

  final FrameData frame;
  final int _rot;

  /// Upright (display-oriented) dimensions.
  late final int width;
  late final int height;

  /// Samples the BGR pixel at upright (x, y), packed as (b << 16)|(g << 8)|r.
  /// Coordinates are clamped to the frame.
  int sampleBgrPacked(int x, int y) {
    if (x < 0) x = 0;
    if (x >= width) x = width - 1;
    if (y < 0) y = 0;
    if (y >= height) y = height - 1;

    final int rawW = frame.width;
    final int rawH = frame.height;
    int rawX, rawY;
    switch (_rot) {
      case 90:
        rawX = y;
        rawY = (width - 1) - x;
        break;
      case 180:
        rawX = (rawW - 1) - x;
        rawY = (rawH - 1) - y;
        break;
      case 270:
        rawX = (height - 1) - y;
        rawY = x;
        break;
      default: // 0
        rawX = x;
        rawY = y;
    }

    if (frame.format == FrameFormat.bgra8888) {
      final int idx = rawY * frame.bgraRowStride + rawX * 4;
      final bytes = frame.bgra!;
      return (bytes[idx] << 16) | (bytes[idx + 1] << 8) | bytes[idx + 2];
    }

    // YUV420 (Android), BT.601.
    final int yIndex = rawY * frame.yRowStride + rawX;
    final int uvIndex =
        (rawY >> 1) * frame.uvRowStride + (rawX >> 1) * frame.uvPixelStride;
    final int yv = frame.yPlane![yIndex];
    final int uv = frame.uPlane![uvIndex] - 128;
    final int vv = frame.vPlane![uvIndex] - 128;
    final int r = (yv + 1.370705 * vv).round().clamp(0, 255);
    final int g = (yv - 0.337633 * uv - 0.698001 * vv).round().clamp(0, 255);
    final int b = (yv + 1.732446 * uv).round().clamp(0, 255);
    return (b << 16) | (g << 8) | r;
  }
}

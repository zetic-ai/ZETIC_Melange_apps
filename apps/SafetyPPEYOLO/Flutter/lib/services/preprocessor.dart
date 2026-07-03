import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Model input is 640x640 (float32[1,3,640,640], NCHW, RGB, 0..1).
const int kInputSize = 640;

/// Pixel formats we know how to decode from the [camera] plugin.
/// Android streams YUV420, iOS streams BGRA8888.
enum FrameFormat { yuv420, bgra8888 }

/// A camera frame flattened into plain typed data. Plane bytes are copied out
/// of the recycled camera buffer inside the image-stream callback.
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
  /// sees an upright scene.
  ///
  /// PyroGuard lesson (CLAUDE.md section 6): on iOS the BGRA buffer arrived
  /// already display-upright (720x1280) and the bug was a SPURIOUS extra
  /// rotation. Do not assume landscape — the camera screen passes 0 for iOS and
  /// the sensor orientation for Android, and the HUD debug line shows the real
  /// buffer WxH so the human can verify at GATE 3.
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
  /// image-stream callback, before the plugin recycles the buffer.
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

/// Result of preprocessing: the NCHW tensor plus the letterbox geometry needed
/// to map model-space boxes back to the original (upright) frame.
class PreprocessResult {
  const PreprocessResult({
    required this.input,
    required this.scale,
    required this.padX,
    required this.padY,
    required this.srcWidth,
    required this.srcHeight,
  });

  /// Flattened float32 buffer, shape [1, 3, 640, 640], NCHW, normalized 0..1.
  /// NOTE: this is a view of the [Preprocessor]'s reusable buffer — valid only
  /// until the next [Preprocessor.run] call.
  final Float32List input;

  /// Upright-source-pixels -> 640 scale factor applied during letterboxing.
  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;
}

/// Fused single-pass letterbox + BILINEAR resample + /255 normalize + NCHW
/// reorder, writing into one pre-allocated buffer (Tier B: no per-frame
/// allocation of the 4.9 MB input tensor).
///
/// Bilinear matters: the Stage-0 accuracy numbers were measured with bilinear
/// resampling (cv2.INTER_LINEAR); shipping nearest-neighbor would silently
/// change model inputs versus the validated harness ("nearest-vs-bilinear
/// nearly shipped" — orchestrator lesson).
class Preprocessor {
  Preprocessor()
      : _input = Float32List(3 * kInputSize * kInputSize);

  final Float32List _input;

  PreprocessResult run(FrameData frame) {
    const int size = kInputSize;
    const int area = size * size;
    final input = _input..fillRange(0, 3 * area, 0.5); // gray letterbox pad

    final int rawW = frame.width;
    final int rawH = frame.height;
    final int rot = ((frame.rotationDegrees % 360) + 360) % 360;
    final bool swap = rot == 90 || rot == 270;

    // Dimensions of the upright (model-facing) frame.
    final int srcW = swap ? rawH : rawW;
    final int srcH = swap ? rawW : rawH;

    final double scale =
        (size / srcW) < (size / srcH) ? (size / srcW) : (size / srcH);
    final int newW = (srcW * scale).round();
    final int newH = (srcH * scale).round();
    final int padX = (size - newW) ~/ 2;
    final int padY = (size - newH) ~/ 2;

    const double inv255 = 1.0 / 255.0;
    final double invScale = 1.0 / scale;

    final bool isBgra = frame.format == FrameFormat.bgra8888;
    final Uint8List? bgra = frame.bgra;
    final Uint8List? yP = frame.yPlane;
    final Uint8List? uP = frame.uPlane;
    final Uint8List? vP = frame.vPlane;

    for (var oy = padY; oy < padY + newH; oy++) {
      // Pixel-center-aligned source y (matches cv2.INTER_LINEAR).
      double fy = (oy - padY + 0.5) * invScale - 0.5;
      if (fy < 0) fy = 0;
      if (fy > srcH - 1) fy = (srcH - 1).toDouble();
      final int y0 = fy.floor();
      final int y1 = y0 + 1 < srcH ? y0 + 1 : y0;
      final double wy = fy - y0;

      final int rowBase = oy * size;
      for (var ox = padX; ox < padX + newW; ox++) {
        double fx = (ox - padX + 0.5) * invScale - 0.5;
        if (fx < 0) fx = 0;
        if (fx > srcW - 1) fx = (srcW - 1).toDouble();
        final int x0 = fx.floor();
        final int x1 = x0 + 1 < srcW ? x0 + 1 : x0;
        final double wx = fx - x0;

        // Bilinear weights over the four upright-space taps.
        final double w00 = (1 - wx) * (1 - wy);
        final double w10 = wx * (1 - wy);
        final double w01 = (1 - wx) * wy;
        final double w11 = wx * wy;

        double r = 0, g = 0, b = 0;
        // Unrolled 4-tap loop; each tap maps upright -> raw coords for `rot`.
        for (var t = 0; t < 4; t++) {
          final int ux = (t == 0 || t == 2) ? x0 : x1;
          final int uy = (t == 0 || t == 1) ? y0 : y1;
          final double w = t == 0
              ? w00
              : t == 1
                  ? w10
                  : t == 2
                      ? w01
                      : w11;
          if (w == 0) continue;

          int rawX, rawY;
          switch (rot) {
            case 90:
              rawX = uy;
              rawY = (srcW - 1) - ux;
              break;
            case 180:
              rawX = (rawW - 1) - ux;
              rawY = (rawH - 1) - uy;
              break;
            case 270:
              rawX = (srcH - 1) - uy;
              rawY = ux;
              break;
            default:
              rawX = ux;
              rawY = uy;
          }

          int tr, tg, tb;
          if (isBgra) {
            final int idx = rawY * frame.bgraRowStride + rawX * 4;
            tb = bgra![idx];
            tg = bgra[idx + 1];
            tr = bgra[idx + 2];
          } else {
            final int yIndex = rawY * frame.yRowStride + rawX;
            final int uvIndex = (rawY >> 1) * frame.uvRowStride +
                (rawX >> 1) * frame.uvPixelStride;
            final int yv = yP![yIndex];
            final int uv = uP![uvIndex] - 128;
            final int vv = vP![uvIndex] - 128;
            // BT.601 YUV -> RGB.
            tr = (yv + 1.370705 * vv).round().clamp(0, 255);
            tg = (yv - 0.337633 * uv - 0.698001 * vv).round().clamp(0, 255);
            tb = (yv + 1.732446 * uv).round().clamp(0, 255);
          }
          r += w * tr;
          g += w * tg;
          b += w * tb;
        }

        final int p = rowBase + ox;
        input[p] = r * inv255;
        input[area + p] = g * inv255;
        input[2 * area + p] = b * inv255;
      }
    }

    return PreprocessResult(
      input: input,
      scale: scale,
      padX: padX,
      padY: padY,
      srcWidth: srcW,
      srcHeight: srcH,
    );
  }
}

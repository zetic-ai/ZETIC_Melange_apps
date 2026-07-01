import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Model input is 640x640.
const int kInputSize = 640;

/// Pixel formats we know how to decode from the [camera] plugin.
/// Android streams YUV420, iOS streams BGRA8888.
enum FrameFormat { yuv420, bgra8888 }

/// A camera frame flattened into plain typed-data so it can be shipped to a
/// background isolate via [compute] (a [CameraImage] itself is not sendable).
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
  /// sees an upright scene. iOS delivers the buffer already upright (0); Android
  /// delivers it in sensor orientation (usually 90) and must be rotated.
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
  factory FrameData.fromCameraImage(CameraImage image, {int rotationDegrees = 0}) {
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
/// to map model-space boxes back to the original frame.
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
  final Float32List input;

  /// Source-pixels -> 640 scale factor applied during letterboxing.
  final double scale;
  final int padX;
  final int padY;
  final int srcWidth;
  final int srcHeight;
}

/// Top-level so it can run in a [compute] isolate.
///
/// Builds the letterboxed, normalized, planar (RGB) input in a single pass over
/// the 640x640 output grid using nearest-neighbor sampling. Padding is filled
/// with the normalized value 0.5.
///
/// If [FrameData.rotationDegrees] is non-zero (Android, sensor-orientation
/// buffer), the source is rotated to upright *during sampling* so the model
/// sees the scene the right way up — critical for detection accuracy, since YOLO
/// is trained on upright images. The returned [PreprocessResult] geometry is in
/// the upright frame, so postprocessing + the overlay need no further rotation.
PreprocessResult preprocessFrame(FrameData frame) {
  const int size = kInputSize;
  const int area = size * size;
  final input = Float32List(3 * area)..fillRange(0, 3 * area, 0.5);

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

  for (var oy = padY; oy < padY + newH; oy++) {
    if (oy < 0 || oy >= size) continue;
    final int uy = ((oy - padY) / scale).floor().clamp(0, srcH - 1);
    final int rowBase = oy * size;
    for (var ox = padX; ox < padX + newW; ox++) {
      if (ox < 0 || ox >= size) continue;
      final int ux = ((ox - padX) / scale).floor().clamp(0, srcW - 1);

      // Map upright (ux, uy) back to raw buffer coords (rawX, rawY).
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
        default: // 0
          rawX = ux;
          rawY = uy;
      }
      if (rawX < 0) rawX = 0;
      if (rawX >= rawW) rawX = rawW - 1;
      if (rawY < 0) rawY = 0;
      if (rawY >= rawH) rawY = rawH - 1;

      int r, g, b;
      if (frame.format == FrameFormat.bgra8888) {
        final int idx = rawY * frame.bgraRowStride + rawX * 4;
        final bytes = frame.bgra!;
        b = bytes[idx];
        g = bytes[idx + 1];
        r = bytes[idx + 2];
      } else {
        final int yIndex = rawY * frame.yRowStride + rawX;
        final int uvIndex = (rawY >> 1) * frame.uvRowStride +
            (rawX >> 1) * frame.uvPixelStride;
        final int yv = frame.yPlane![yIndex];
        final int uv = frame.uPlane![uvIndex] - 128;
        final int vv = frame.vPlane![uvIndex] - 128;
        // BT.601 YUV -> RGB.
        r = (yv + 1.370705 * vv).round().clamp(0, 255);
        g = (yv - 0.337633 * uv - 0.698001 * vv).round().clamp(0, 255);
        b = (yv + 1.732446 * uv).round().clamp(0, 255);
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

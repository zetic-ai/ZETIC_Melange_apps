import 'dart:typed_data';

import 'package:camera/camera.dart';

/// Pixel formats we know how to decode from the [camera] plugin.
/// Android streams YUV420, iOS streams BGRA8888.
enum FrameFormat { yuv420, bgra8888 }

/// A camera frame flattened into plain typed-data so it can be shipped to a
/// background isolate via [compute] (a [CameraImage] itself is not sendable).
///
/// Copied out of the recycled camera buffer inside the image-stream callback,
/// then owned by this object and safe to send across isolates.
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
  /// (and ML Kit) see an upright scene. Must match the rotation handed to ML
  /// Kit's [InputImage] so the detector's coordinates and this buffer share one
  /// upright space. iOS typically delivers the buffer already upright (0);
  /// Android delivers it in sensor orientation (usually 90). Measure on-device.
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

  factory FrameData.fromCameraImage(
    CameraImage image, {
    int rotationDegrees = 0,
  }) {
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

/// An upright, interleaved **BGR** image (3 bytes/pixel, B,G,R order), in the
/// same coordinate space as the ML Kit face box / landmarks. Both Melange
/// branches sample from this: BGR is what PAD and SFace expect, and building it
/// once per frame avoids re-decoding the raw planes twice.
class BgrImage {
  BgrImage({
    required this.bgr,
    required this.width,
    required this.height,
  });

  /// Interleaved BGR, length == width * height * 3.
  final Uint8List bgr;
  final int width;
  final int height;

  /// Reads the blue/green/red bytes at an integer pixel, clamped to bounds.
  /// Returned as a 3-int list [b, g, r].
  int _index(int x, int y) {
    var cx = x;
    var cy = y;
    if (cx < 0) cx = 0;
    if (cx >= width) cx = width - 1;
    if (cy < 0) cy = 0;
    if (cy >= height) cy = height - 1;
    return (cy * width + cx) * 3;
  }

  int blueAt(int x, int y) => bgr[_index(x, y)];
  int greenAt(int x, int y) => bgr[_index(x, y) + 1];
  int redAt(int x, int y) => bgr[_index(x, y) + 2];
}

/// Converts a raw camera [FrameData] into an upright [BgrImage].
///
/// Rotation is applied so the returned image is display-upright and matches the
/// rotation metadata handed to ML Kit. No horizontal mirror is applied even for
/// the front camera: ML Kit reports coordinates in the un-mirrored rotated
/// image, so the model path stays un-mirrored too and the two spaces agree.
/// (Preview mirroring is a pure UI concern handled in the widget layer.)
///
/// Top-level so it can run inside a [compute] isolate.
BgrImage frameToUprightBgr(FrameData frame) {
  final int rawW = frame.width;
  final int rawH = frame.height;
  final int rot = ((frame.rotationDegrees % 360) + 360) % 360;
  final bool swap = rot == 90 || rot == 270;
  final int outW = swap ? rawH : rawW;
  final int outH = swap ? rawW : rawH;

  final out = Uint8List(outW * outH * 3);

  for (var oy = 0; oy < outH; oy++) {
    for (var ox = 0; ox < outW; ox++) {
      // Map upright (ox, oy) back to raw buffer coords.
      int rawX, rawY;
      switch (rot) {
        case 90:
          rawX = oy;
          rawY = (outW - 1) - ox;
          break;
        case 180:
          rawX = (rawW - 1) - ox;
          rawY = (rawH - 1) - oy;
          break;
        case 270:
          rawX = (outH - 1) - oy;
          rawY = ox;
          break;
        default: // 0
          rawX = ox;
          rawY = oy;
      }
      if (rawX < 0) rawX = 0;
      if (rawX >= rawW) rawX = rawW - 1;
      if (rawY < 0) rawY = 0;
      if (rawY >= rawH) rawY = rawH - 1;

      int b, g, r;
      if (frame.format == FrameFormat.bgra8888) {
        final int idx = rawY * frame.bgraRowStride + rawX * 4;
        final bytes = frame.bgra!;
        b = bytes[idx];
        g = bytes[idx + 1];
        r = bytes[idx + 2];
      } else {
        final int yIndex = rawY * frame.yRowStride + rawX;
        final int uvIndex =
            (rawY >> 1) * frame.uvRowStride + (rawX >> 1) * frame.uvPixelStride;
        final int yv = frame.yPlane![yIndex];
        final int uv = frame.uPlane![uvIndex] - 128;
        final int vv = frame.vPlane![uvIndex] - 128;
        // BT.601 YUV -> RGB.
        r = (yv + 1.370705 * vv).round().clamp(0, 255);
        g = (yv - 0.337633 * uv - 0.698001 * vv).round().clamp(0, 255);
        b = (yv + 1.732446 * uv).round().clamp(0, 255);
      }

      final int p = (oy * outW + ox) * 3;
      out[p] = b;
      out[p + 1] = g;
      out[p + 2] = r;
    }
  }

  return BgrImage(bgr: out, width: outW, height: outH);
}

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

  /// Clockwise rotation (0/90/180/270) that uprights the raw buffer.
  ///
  /// Measured, never assumed: on the PyroGuard iOS setup the BGRA buffer
  /// arrived already upright (rotation 0) and the historical bug was a
  /// *spurious* rotation. Android delivers sensor orientation (usually 90).
  /// The actual buffer WxH is surfaced on the HUD for on-device verification.
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

/// Maps an upright-frame pixel coordinate back to raw-buffer coordinates for
/// clockwise buffer rotation [rot] (0/90/180/270). `rawW`/`rawH` are the RAW
/// buffer dimensions. Exposed for the orientation round-trip tests.
(int, int) uprightToRaw(int ux, int uy, int rawW, int rawH, int rot) {
  switch (((rot % 360) + 360) % 360) {
    case 90:
      // Upright frame is rawH wide, rawW tall.
      return (uy, rawW - 1 - ux);
    case 180:
      return (rawW - 1 - ux, rawH - 1 - uy);
    case 270:
      return (rawH - 1 - uy, ux);
    default:
      return (ux, uy);
  }
}

/// Exact inverse of [uprightToRaw] — a raw-buffer pixel to upright-frame
/// coordinates. Used only by tests to prove the transform round-trips.
(int, int) rawToUpright(int rx, int ry, int rawW, int rawH, int rot) {
  switch (((rot % 360) + 360) % 360) {
    case 90:
      return (rawW - 1 - ry, rx);
    case 180:
      return (rawW - 1 - rx, rawH - 1 - ry);
    case 270:
      return (ry, rawH - 1 - rx);
    default:
      return (rx, ry);
  }
}

import 'dart:typed_data';

/// Camera pixel formats we accept. iOS delivers BGRA8888; Android CameraX
/// delivers YUV420 (the cheapest usable formats on each platform).
enum FramePixelFormat { bgra8888, yuv420 }

/// A single raw camera frame, packaged so it can be sent *into* the long-lived
/// inference isolate with one copy. Holds only plain typed data + ints.
class FrameData {
  const FrameData({
    required this.format,
    required this.width,
    required this.height,
    required this.rotationDegrees,
    required this.plane0,
    required this.bytesPerRow0,
    this.plane1,
    this.plane2,
    this.bytesPerRow1 = 0,
    this.bytesPerRow2 = 0,
    this.pixelStride1 = 1,
    this.pixelStride2 = 1,
  });

  final FramePixelFormat format;

  /// Raw buffer dimensions (before any upright rotation).
  final int width;
  final int height;

  /// Clockwise rotation to make the buffer upright for the model.
  final int rotationDegrees;

  /// BGRA: the whole image. YUV420: the Y plane.
  final Uint8List plane0;
  final int bytesPerRow0;

  /// YUV420 chroma planes (U=plane1, V=plane2). Null for BGRA.
  final Uint8List? plane1;
  final Uint8List? plane2;
  final int bytesPerRow1;
  final int bytesPerRow2;
  final int pixelStride1;
  final int pixelStride2;
}

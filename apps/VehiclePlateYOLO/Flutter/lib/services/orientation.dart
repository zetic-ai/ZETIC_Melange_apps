/// Maps between the raw camera *buffer* pixel grid and the *upright* image the
/// model expects. YOLO is trained on upright frames; feeding a sideways buffer
/// craters confidence (the PyroGuard Android bug). On the reference iOS BGRA
/// setup the buffer arrives already upright (720x1280) so rotation is 0 and
/// every transform here is identity — the spec's measured default. Android
/// YUV420 is sensor-landscape and needs 90 (Tier C device calibration).
///
/// Pure Dart, isolate-safe, and round-trippable so VALIDATION.md A3
/// "orientation" can assert the chosen transform returns a known box.
class FrameOrientation {
  const FrameOrientation(this.rotationDegrees)
    : assert(
        rotationDegrees == 0 ||
            rotationDegrees == 90 ||
            rotationDegrees == 180 ||
            rotationDegrees == 270,
        'rotation must be a right angle',
      );

  /// Clockwise rotation (degrees) applied to the buffer to make it upright.
  final int rotationDegrees;

  bool get isIdentity => rotationDegrees == 0;

  /// Upright image width given the raw buffer dimensions.
  int uprightWidth(int bufW, int bufH) =>
      (rotationDegrees == 90 || rotationDegrees == 270) ? bufH : bufW;

  /// Upright image height given the raw buffer dimensions.
  int uprightHeight(int bufW, int bufH) =>
      (rotationDegrees == 90 || rotationDegrees == 270) ? bufW : bufH;

  /// Map an upright-image pixel (ux,uy) back to the raw buffer pixel (bx,by).
  /// Used during preprocessing sampling. Inverse of [bufferToUpright].
  List<int> uprightToBuffer(int ux, int uy, int bufW, int bufH) {
    switch (rotationDegrees) {
      case 90:
        // upright (ux,uy) came from buffer rotated 90 CW.
        return [uy, bufH - 1 - ux];
      case 180:
        return [bufW - 1 - ux, bufH - 1 - uy];
      case 270:
        return [bufW - 1 - uy, ux];
      default:
        return [ux, uy];
    }
  }

  /// Map a raw buffer pixel (bx,by) to the upright-image pixel (ux,uy).
  /// Inverse of [uprightToBuffer]; used only by the round-trip test.
  List<int> bufferToUpright(int bx, int by, int bufW, int bufH) {
    switch (rotationDegrees) {
      case 90:
        return [bufH - 1 - by, bx];
      case 180:
        return [bufW - 1 - bx, bufH - 1 - by];
      case 270:
        return [by, bufW - 1 - bx];
      default:
        return [bx, by];
    }
  }
}

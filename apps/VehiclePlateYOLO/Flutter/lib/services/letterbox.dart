import 'dart:math' as math;

/// Letterbox geometry for a fixed square model input (default 640x640).
///
/// Forward: source upright-image pixel space -> letterboxed model space.
/// Inverse: letterboxed model space -> source upright-image pixel space.
///
/// The inverse is the *exact* reverse of the forward steps, which is the trap
/// VALIDATION.md A3 "letterbox / resize inverse" guards against. Pure Dart, no
/// Flutter import, so it is unit-testable and isolate-safe.
class LetterboxParams {
  const LetterboxParams({
    required this.srcWidth,
    required this.srcHeight,
    required this.target,
    required this.scale,
    required this.padX,
    required this.padY,
  });

  /// Upright source-image dimensions (after any rotation has been applied).
  final int srcWidth;
  final int srcHeight;

  /// Square model side (e.g. 640).
  final int target;

  /// Uniform scale = min(target/srcW, target/srcH) (aspect preserved).
  final double scale;

  /// Symmetric padding (pixels) in model space on each axis.
  final double padX;
  final double padY;

  factory LetterboxParams.forImage(int srcWidth, int srcHeight, int target) {
    final scale = math.min(target / srcWidth, target / srcHeight);
    final scaledW = srcWidth * scale;
    final scaledH = srcHeight * scale;
    final padX = (target - scaledW) / 2.0;
    final padY = (target - scaledH) / 2.0;
    return LetterboxParams(
      srcWidth: srcWidth,
      srcHeight: srcHeight,
      target: target,
      scale: scale,
      padX: padX,
      padY: padY,
    );
  }

  /// source-image x -> model-space x.
  double forwardX(double x) => x * scale + padX;

  /// source-image y -> model-space y.
  double forwardY(double y) => y * scale + padY;

  /// model-space x -> source-image x (exact inverse of [forwardX]).
  double inverseX(double x) => (x - padX) / scale;

  /// model-space y -> source-image y (exact inverse of [forwardY]).
  double inverseY(double y) => (y - padY) / scale;
}

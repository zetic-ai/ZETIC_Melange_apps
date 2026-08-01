import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/letterbox.dart';

/// TRAP: the letterbox inverse must be the EXACT reverse of the forward steps.
/// A wrong pad or scale shifts every box. Round-trip a known box on a non-square
/// source (so scale and pad are both non-trivial).
void main() {
  test('forward then inverse returns box within tolerance (non-square src)', () {
    // iOS-style upright buffer.
    final params = LetterboxParams.forImage(720, 1280, 640);

    // scale = min(640/720, 640/1280) = 0.5 ; padX = (640-360)/2 = 140 ; padY = 0
    expect(params.scale, closeTo(0.5, 1e-9));
    expect(params.padX, closeTo(140, 1e-9));
    expect(params.padY, closeTo(0, 1e-9));

    for (final box in [
      [100.0, 200.0],
      [260.0, 360.0],
      [0.0, 0.0],
      [719.0, 1279.0],
    ]) {
      final fx = params.forwardX(box[0]);
      final fy = params.forwardY(box[1]);
      expect(params.inverseX(fx), closeTo(box[0], 1e-6));
      expect(params.inverseY(fy), closeTo(box[1], 1e-6));
    }
  });
}

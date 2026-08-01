import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/frame_data.dart';
import 'package:vehicleplateyolo/services/preprocessor.dart';

/// TRAP: channel order (BGRA source -> RGB), NCHW vs NHWC packing, and the /255
/// normalize. A wrong channel order or layout silently feeds the model garbage.
/// Use a tiny 2x2 frame and target=2 so the letterbox is identity and we can
/// assert every packed value.
void main() {
  test('BGRA->RGB, NCHW planar packing, /255 normalize', () {
    // Distinct RGB per pixel, stored as BGRA bytes [B,G,R,A].
    // (0,0)=R10 G20 B30  (1,0)=R40 G50 B60  (0,1)=R70 G80 B90  (1,1)=R100 G110 B120
    final bgra = Uint8List.fromList([
      30, 20, 10, 255, /* (0,0) */ 60, 50, 40, 255, /* (1,0) */
      90, 80, 70, 255, /* (0,1) */ 120, 110, 100, 255, /* (1,1) */
    ]);

    final frame = FrameData(
      format: FramePixelFormat.bgra8888,
      width: 2,
      height: 2,
      rotationDegrees: 0,
      plane0: bgra,
      bytesPerRow0: 8,
    );

    final pre = Preprocessor(target: 2).process(frame);
    final b = pre.input;
    const plane = 2 * 2;

    // R plane (channel 0): pixels in row-major order.
    expect(b[0], closeTo(10 / 255, 1e-6));
    expect(b[1], closeTo(40 / 255, 1e-6));
    expect(b[2], closeTo(70 / 255, 1e-6));
    expect(b[3], closeTo(100 / 255, 1e-6));
    // G plane (channel 1).
    expect(b[plane + 0], closeTo(20 / 255, 1e-6));
    expect(b[plane + 3], closeTo(110 / 255, 1e-6));
    // B plane (channel 2).
    expect(b[2 * plane + 0], closeTo(30 / 255, 1e-6));
    expect(b[2 * plane + 3], closeTo(120 / 255, 1e-6));

    // Identity letterbox for a square source.
    expect(pre.params.scale, closeTo(1.0, 1e-9));
    expect(pre.params.padX, closeTo(0, 1e-9));
    expect(pre.params.padY, closeTo(0, 1e-9));
  });
}

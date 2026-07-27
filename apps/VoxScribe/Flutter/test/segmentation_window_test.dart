import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/preprocessor.dart';

/// A4 — segmentation window framing. Pad/truncate to exactly 160000 samples.
void main() {
  test('short input zero-pads tail to exactly 160000', () {
    final Float32List sig = Float32List(80000)
      ..fillRange(0, 80000, 0.5); // 5 s of 0.5
    final Float32List win = segmentationWindow(sig);
    expect(win.length, kSegmentationSamples); // 160000
    expect(win[0], closeTo(0.5, 1e-6));
    expect(win[79999], closeTo(0.5, 1e-6));
    // Pad region is zero.
    expect(win[80000], 0.0);
    expect(win[159999], 0.0);
  });

  test('long input truncates to 160000', () {
    final Float32List sig = Float32List(200000)..fillRange(0, 200000, 0.3);
    final Float32List win = segmentationWindow(sig);
    expect(win.length, kSegmentationSamples);
    expect(win[159999], closeTo(0.3, 1e-6));
  });
}

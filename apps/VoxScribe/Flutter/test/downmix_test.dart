import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/preprocessor.dart';

/// A2 — mono downmix trap. Reference uses channel 0 (SPEC), NOT an average.
void main() {
  test('interleaved stereo downmixes to channel 0, length halved', () {
    // [L0,R0,L1,R1,L2,R2] with distinct L/R so an average would differ.
    final Float32List stereo = Float32List.fromList(<double>[
      0.10, 0.90, // frame 0
      0.20, 0.80, // frame 1
      0.30, 0.70, // frame 2
    ]);
    final Float32List mono = downmixCh0(stereo, 2);
    expect(mono.length, 3);
    expect(mono[0], closeTo(0.10, 1e-6)); // ch0, not (0.1+0.9)/2
    expect(mono[1], closeTo(0.20, 1e-6));
    expect(mono[2], closeTo(0.30, 1e-6));
  });

  test('mono input passes through unchanged', () {
    final Float32List mono = Float32List.fromList(<double>[0.1, 0.2, 0.3]);
    expect(downmixCh0(mono, 1), orderedEquals(mono));
  });
}

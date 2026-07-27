import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/preprocessor.dart';

/// A1 — sample-rate trap. N input seconds must map to N output seconds.
void main() {
  test('48 kHz ramp (1 s) -> 16 kHz yields 16000 samples, endpoints preserved',
      () {
    final Float32List ramp = Float32List(48000);
    for (int i = 0; i < ramp.length; i++) {
      ramp[i] = i / 47999.0; // 0..1 ramp
    }
    final Float32List out = resampleLinear(ramp, 48000, 16000);
    expect(out.length, 16000); // 1 s in -> 1 s out
    expect(out.first, closeTo(0.0, 1e-6));
    expect(out.last, closeTo(1.0, 1e-6));
    // Midpoint should be ~0.5 for a linear ramp.
    expect(out[8000], closeTo(0.5, 1e-3));
  });

  test('44.1 kHz (2 s) -> 16 kHz yields 32000 samples', () {
    final Float32List sig = Float32List(88200); // 2 s @ 44.1 kHz
    final Float32List out = resampleLinear(sig, 44100, 16000);
    expect(out.length, 32000); // 2 s
  });

  test('16 kHz input is a no-op (identity copy)', () {
    final Float32List sig = Float32List.fromList(<double>[0.1, -0.2, 0.3]);
    final Float32List out = resampleLinear(sig, 16000, 16000);
    expect(out, orderedEquals(sig));
  });
}

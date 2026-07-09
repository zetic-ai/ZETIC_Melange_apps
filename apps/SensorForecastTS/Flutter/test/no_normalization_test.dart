import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/services/preprocessor.dart';

void main() {
  group('NO-normalization contract (in-graph instance norm)', () {
    test('window snapshot passes raw values bit-exact (float32)', () {
      final w = SampleWindow();
      final values = List<double>.generate(
          kContextLength, (i) => 100.0 + i * 0.657 - (i % 13) * 3.21);
      for (final v in values) {
        w.push(v);
      }
      final dst = Float32List(kContextLength);
      w.snapshotInto(dst);
      for (var i = 0; i < kContextLength; i++) {
        // Exactly the float32 cast of the input: no /255, no z-score, no
        // offset, no clipping. (Bit-exact within the float32 round of push.)
        final expected = Float32List(1)..[0] = values[i];
        expect(dst[i], expected[0],
            reason: 'sample $i must be untouched raw value');
      }
    });

    test('values in the hundreds survive untouched (no 0-1 squash)', () {
      final w = SampleWindow();
      for (var i = 0; i < kContextLength; i++) {
        w.push(880.5);
      }
      final dst = Float32List(kContextLength);
      w.snapshotInto(dst);
      expect(dst.every((v) => v == 880.5), isTrue);
    });

    test('negative and tiny values survive untouched', () {
      final w = SampleWindow();
      for (var i = 0; i < kContextLength; i++) {
        w.push(i.isEven ? -40.25 : 0.001953125); // exact float32 values
      }
      final dst = Float32List(kContextLength);
      w.snapshotInto(dst);
      expect(dst[0], -40.25);
      expect(dst[1], 0.001953125);
    });
  });
}

import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sensorforecastts/services/preprocessor.dart';

void main() {
  group('SampleWindow (ring buffer)', () {
    test('tensor is exactly the LAST 512 samples, oldest -> newest', () {
      final w = SampleWindow();
      // 600 known samples: value == its global index.
      for (var i = 0; i < 600; i++) {
        w.push(i.toDouble());
      }
      final dst = Float32List(kContextLength);
      w.snapshotInto(dst);
      // Last 512 of 600 are indices 88..599, in that order.
      expect(dst.first, 88.0);
      expect(dst.last, 599.0);
      for (var k = 0; k < kContextLength; k++) {
        expect(dst[k], (88 + k).toDouble(),
            reason: 'position $k must be global sample ${88 + k}');
      }
    });

    test('order is oldest->newest, not reversed', () {
      final w = SampleWindow();
      for (var i = 0; i < kContextLength; i++) {
        w.push(i.toDouble());
      }
      final dst = Float32List(kContextLength);
      w.snapshotInto(dst);
      expect(dst[0], lessThan(dst[kContextLength - 1]));
      expect(dst[0], 0.0);
      expect(dst[511], 511.0);
    });

    test('exact-capacity boundary (no off-by-one)', () {
      final w = SampleWindow();
      for (var i = 0; i < kContextLength - 1; i++) {
        w.push(1.0);
      }
      expect(w.isFull, isFalse, reason: '511 samples is NOT full');
      w.push(2.0);
      expect(w.isFull, isTrue, reason: '512th sample fills the window');
    });

    test('FULL-WINDOW CONTRACT: snapshot refused before 512 real samples', () {
      final w = SampleWindow();
      for (var i = 0; i < 100; i++) {
        w.push(1.0);
      }
      final dst = Float32List(kContextLength);
      expect(() => w.snapshotInto(dst), throwsStateError,
          reason: 'inference must never run on a partial window');
    });

    test('wrap-around keeps ordering across many laps', () {
      final w = SampleWindow(4);
      for (var i = 0; i < 11; i++) {
        w.push(i.toDouble());
      }
      final dst = Float32List(4);
      w.snapshotInto(dst);
      expect(dst, orderedEquals(const [7.0, 8.0, 9.0, 10.0]));
    });
  });
}

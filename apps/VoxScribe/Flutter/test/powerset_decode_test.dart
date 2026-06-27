import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/postprocessor.dart';

/// A6 — 7-class powerset decode (incl. overlaps 4/5/6). 7 classes != 7 speakers.
void main() {
  test('each class maps to the correct LOCAL-speaker set', () {
    const int n = 7;
    final Float32List logits = Float32List(n * kSegClasses)
      ..fillRange(0, n * kSegClasses, -10.0);
    // frame f -> argmax class f
    for (int f = 0; f < n; f++) {
      logits[f * kSegClasses + f] = 5.0;
    }
    final List<List<bool>> labels = powersetDecode(logits, nFrames: n);

    bool spk(int f, int s) => labels[f][s];
    // 0:{} 1:{0} 2:{1} 3:{2} 4:{0,1} 5:{0,2} 6:{1,2}
    expect(<bool>[spk(0, 0), spk(0, 1), spk(0, 2)], <bool>[false, false, false]);
    expect(<bool>[spk(1, 0), spk(1, 1), spk(1, 2)], <bool>[true, false, false]);
    expect(<bool>[spk(2, 0), spk(2, 1), spk(2, 2)], <bool>[false, true, false]);
    expect(<bool>[spk(3, 0), spk(3, 1), spk(3, 2)], <bool>[false, false, true]);
    expect(<bool>[spk(4, 0), spk(4, 1), spk(4, 2)], <bool>[true, true, false]);
    expect(<bool>[spk(5, 0), spk(5, 1), spk(5, 2)], <bool>[true, false, true]);
    expect(<bool>[spk(6, 0), spk(6, 1), spk(6, 2)], <bool>[false, true, true]);
  });

  test('decode table constant matches the SPEC', () {
    expect(kPowersetTable[0], <int>[]);
    expect(kPowersetTable[4], <int>[0, 1]);
    expect(kPowersetTable[5], <int>[0, 2]);
    expect(kPowersetTable[6], <int>[1, 2]);
  });
}

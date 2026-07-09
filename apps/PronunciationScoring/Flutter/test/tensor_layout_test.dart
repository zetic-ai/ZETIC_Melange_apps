import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:sayright/models/phonemes.dart';
import 'package:sayright/services/postprocessor.dart';

void main() {
  const frames = Phonemes.frameCount; // 64
  const classes = Phonemes.classCount; // 45

  // Build a flat buffer with ONE known argmax per frame, placed frame-major:
  // index = frame * classes + class. Each frame's peak class is unique-ish.
  int peakClassFor(int f) => (f * 7) % classes;

  Float32List buildFrameMajor() {
    final data = Float32List(frames * classes);
    for (var f = 0; f < frames; f++) {
      for (var c = 0; c < classes; c++) {
        data[f * classes + c] = -10.0;
      }
      data[f * classes + peakClassFor(f)] = 5.0 + f; // distinct peak
    }
    return data;
  }

  group('tensor layout is frame-major [1,64,45]', () {
    test('argmaxAt recovers the frame-major peak class', () {
      final lp = LogProbView(buildFrameMajor());
      for (var f = 0; f < frames; f++) {
        expect(lp.argmaxAt(f), peakClassFor(f), reason: 'frame $f');
      }
    });

    test('at(frame, class) indexes frame*45 + class', () {
      final data = buildFrameMajor();
      final lp = LogProbView(data);
      expect(lp.at(3, peakClassFor(3)), data[3 * classes + peakClassFor(3)]);
      expect(lp.at(3, peakClassFor(3)), 5.0 + 3);
    });

    test('reading the SAME buffer class-major gives a different (wrong) answer',
        () {
      final data = buildFrameMajor();
      // Class-major misread: treat index as class*frames + frame.
      int classMajorArgmax(int frame) {
        var best = 0;
        var bestVal = data[0 * frames + frame];
        for (var c = 1; c < classes; c++) {
          final v = data[c * frames + frame];
          if (v > bestVal) {
            bestVal = v;
            best = c;
          }
        }
        return best;
      }

      final lp = LogProbView(data);
      var disagreements = 0;
      for (var f = 0; f < frames; f++) {
        if (classMajorArgmax(f) != lp.argmaxAt(f)) disagreements++;
      }
      // If layout didn't matter the two readings would always agree.
      expect(disagreements, greaterThan(0),
          reason: 'class-major misread must diverge from frame-major truth');
    });
  });
}

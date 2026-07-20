import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/services/ctc_decoder.dart';

void main() {
  // A synthetic 838-class map with distinct single characters at known
  // indices, sized exactly like production (blank@0, 836 chars, space@837).
  final lines = List.generate(836, (i) => String.fromCharCode(0x100 + i));
  final decoder = CtcDecoder.fromCharsetLines(lines);

  group('time-major [1,40,838] layout', () {
    test('decode steps over T=40 with argmax over the LAST axis (C=838)', () {
      // Adversarial construction: write the tensor flat so that the CORRECT
      // (time-major) read finds class (t+1) at step t for the first 3 steps,
      // while a TRANSPOSED ([838,40]) read of the same buffer would see a
      // completely different argmax pattern.
      final out = Float32List(kRecTimeSteps * kRecNumClasses);
      // Correct layout: probs[t * 838 + c].
      out[0 * kRecNumClasses + 1] = 0.9; // step 0 -> class 1
      out[1 * kRecNumClasses + 2] = 0.8; // step 1 -> class 2
      out[2 * kRecNumClasses + 3] = 0.7; // step 2 -> class 3
      // Poison the transposed interpretation: under a [C,T] read, element
      // [c=1,t=0] would sit at flat index 1*40+0=40 — plant a big value at
      // flat 40 that the CORRECT read sees as step 0, class 40 (weaker than
      // 0.9 so it must NOT win if the axis order is right).
      out[40] = 0.5;

      final result = decoder.decode(out);
      expect(result.text,
          '${decoder.classes[1]}${decoder.classes[2]}${decoder.classes[3]}');
    });

    test('a transposed read would produce different output (trap is live)',
        () {
      // Same construction, decoded deliberately with swapped axis roles —
      // proving the test would FAIL under the transposed interpretation
      // (i.e. this trap actually discriminates).
      final out = Float32List(kRecTimeSteps * kRecNumClasses);
      out[0 * kRecNumClasses + 1] = 0.9;
      out[1 * kRecNumClasses + 2] = 0.8;

      // Transposed decode: step t reads column t of an [838,40] matrix.
      final transposedFirstStepArgmax = () {
        var best = 0;
        var bestV = -1.0;
        for (var c = 0; c < kRecNumClasses; c++) {
          // element [c, t=0] under transposed layout = flat c*40+0
          final flat = c * kRecTimeSteps;
          if (flat < out.length && out[flat] > bestV) {
            bestV = out[flat];
            best = c;
          }
        }
        return best;
      }();

      // The correct read finds class 1 at step 0; the transposed read does
      // not (0.9 lives at flat 839, which is NOT a multiple of 40).
      expect(transposedFirstStepArgmax, isNot(1));
    });
  });

  group('no extra softmax (baked into the ONNX)', () {
    test('values are used as probabilities directly — no renormalization',
        () {
      // A realistic already-softmaxed step: row sums to 1, max prob 0.75.
      final out = Float32List(kRecTimeSteps * kRecNumClasses);
      final uniform = 0.25 / (kRecNumClasses - 1);
      for (var c = 0; c < kRecNumClasses; c++) {
        out[c] = c == 5 ? 0.75 : uniform;
      }
      // Remaining steps: blank at 0.9.
      for (var t = 1; t < kRecTimeSteps; t++) {
        out[t * kRecNumClasses] = 0.9;
      }

      final result = decoder.decode(out);
      expect(result.text, decoder.classes[5]);
      // Confidence must be EXACTLY the raw 0.75 — any softmax/sigmoid layer
      // in Dart would distort it (softmax(0.75 vs floor) ≈ tiny).
      expect(result.confidence, closeTo(0.75, 1e-6));
    });
  });
}

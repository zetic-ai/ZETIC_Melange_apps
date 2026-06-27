import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/postprocessor.dart';

/// A7 — log-softmax semantics. The segmentation output is log-softmax: exp(row)
/// sums to 1, argmax is invariant to exp, and any probability threshold must
/// exp() first (compare against exp(value), not the raw log value).
void main() {
  test('exp(log-softmax row) sums to 1 and argmax is exp-invariant', () {
    final List<double> rawLogits = <double>[2.0, -1.0, 0.5, 3.5, -0.2, 1.0, 0.0];
    final double maxL = rawLogits.reduce(math.max);
    double sumExp = 0;
    for (final double l in rawLogits) {
      sumExp += math.exp(l - maxL);
    }
    final double logSumExp = maxL + math.log(sumExp);
    final Float32List logSoftmax = Float32List.fromList(
      rawLogits.map((double l) => l - logSumExp).toList(),
    );

    // exp sums to 1.
    double probSum = 0;
    for (final double v in logSoftmax) {
      probSum += math.exp(v);
    }
    expect(probSum, closeTo(1.0, 1e-6));

    // argmax invariant under exp (so powersetDecode can argmax the raw row).
    final int argLog = argmaxRange(logSoftmax, 0, logSoftmax.length);
    final Float32List probs =
        Float32List.fromList(logSoftmax.map(math.exp).toList());
    final int argProb = argmaxRange(probs, 0, probs.length);
    expect(argLog, argProb);
    expect(argLog, 3); // the 3.5 logit

    // A 0.5 probability threshold must exp() the log value first.
    final double topProb = math.exp(logSoftmax[3]);
    expect(topProb, greaterThan(0.5));
    expect(logSoftmax[3], lessThan(0.0)); // log-prob is negative, would fail raw
  });
}

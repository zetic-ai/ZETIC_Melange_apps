import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/postprocessor.dart';

/// A5 — segmentation frame->time map. 589 frames; frame*0.016875 + 0.0309688.
void main() {
  test('frame constants and time map', () {
    expect(kSegFrames, 589);
    expect(kFrameScale, closeTo(270.0 / 16000.0, 1e-12)); // 0.016875
    expect(kFrameOffset, closeTo(0.0309688, 1e-6));

    expect(frameToTime(0), closeTo(kFrameOffset, 1e-12));
    expect(frameToTime(100), closeTo(100 * kFrameScale + kFrameOffset, 1e-12));
    expect(frameToTime(588), closeTo(588 * kFrameScale + kFrameOffset, 1e-12));
  });
}

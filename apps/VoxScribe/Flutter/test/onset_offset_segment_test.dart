import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/models/speaker_segment.dart';
import 'package:voxscribe/services/postprocessor.dart';

/// A8 — onset/offset segmentation state machine (min_duration_on/off).
/// Builds binary activity with: a short intra-gap that should MERGE, a long gap
/// that should SEPARATE, and a sub-0.30 s blip that should be DROPPED.
void main() {
  test('merges <=0.50 s gaps, splits >0.50 s gaps, drops <0.30 s blips', () {
    const int n = 205;
    final List<List<bool>> labels = List<List<bool>>.generate(
      n,
      (_) => <bool>[false, false, false],
      growable: false,
    );
    void activate(int from, int toExcl) {
      for (int f = from; f < toExcl; f++) {
        labels[f][0] = true; // local speaker 0
      }
    }

    activate(0, 31); // run 1a (~0.52 s)
    // gap 31..39 (~0.135 s) -> merges
    activate(39, 51); // run 1b -> merged with 1a => [0, 51)
    // gap 51..121 (~1.18 s) -> separates
    activate(121, 161); // run 2 (~0.675 s)
    // gap 161..196 (~0.59 s) -> separates
    activate(196, 200); // blip (~0.067 s) -> dropped

    final List<SpeakerSegment> segs = onsetOffsetSegments(labels);

    expect(segs.length, 2); // blip dropped, intra-gap merged
    expect(segs.every((SpeakerSegment s) => s.speaker == 0), isTrue);

    expect(segs[0].start, closeTo(frameToTime(0), 1e-9));
    expect(segs[0].end, closeTo(frameToTime(51), 1e-9));
    expect(segs[1].start, closeTo(frameToTime(121), 1e-9));
    expect(segs[1].end, closeTo(frameToTime(161), 1e-9));
  });
}

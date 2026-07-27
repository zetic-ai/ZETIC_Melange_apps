import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/models/speaker_segment.dart';
import 'package:voxscribe/models/transcript_line.dart';
import 'package:voxscribe/services/diarization_fusion.dart';

/// A14 — fusion attribution. Each span's text is tagged with the segment's
/// speaker BY CONSTRUCTION (diarize-then-transcribe). Two segments + a stub
/// transcriber -> two attributed lines in timeline order.
void main() {
  test('two segments map to two speaker-tagged lines in order', () {
    final List<SpeakerSegment> segs = <SpeakerSegment>[
      const SpeakerSegment(start: 0.0, end: 2.0, speaker: 0),
      const SpeakerSegment(start: 2.0, end: 4.0, speaker: 1),
    ];
    final Map<int, String> stub = <int, String>{0: 'hello', 1: 'world'};

    final List<TranscriptLine> lines =
        fuse(segs, (SpeakerSegment s) => stub[s.speaker]!);

    expect(lines.length, 2);
    expect(lines[0].speaker, 1); // 1-based label for local slot 0
    expect(lines[0].text, 'hello');
    expect(lines[0].start, 0.0);
    expect(lines[0].end, 2.0);
    expect(lines[1].speaker, 2); // local slot 1
    expect(lines[1].text, 'world');
    expect(lines[1].start, 2.0);
    expect(lines[1].end, 4.0);
  });

  test('empty-text spans are dropped; output stays timeline-ordered', () {
    final List<SpeakerSegment> segs = <SpeakerSegment>[
      const SpeakerSegment(start: 2.0, end: 4.0, speaker: 1),
      const SpeakerSegment(start: 0.0, end: 2.0, speaker: 0), // out of order
    ];
    final List<TranscriptLine> lines = fuse(
      segs,
      (SpeakerSegment s) => s.speaker == 0 ? '' : 'kept',
    );
    expect(lines.length, 1);
    expect(lines[0].text, 'kept');
    expect(lines[0].speaker, 2);
  });
}

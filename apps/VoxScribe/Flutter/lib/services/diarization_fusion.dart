import '../models/speaker_segment.dart';
import '../models/transcript_line.dart';

/// Transcribes one speaker span -> text. The real implementation runs the
/// Whisper encoder+decoder over the span; tests inject a stub.
typedef SpanTranscriber = String Function(SpeakerSegment segment);

/// Diarize-THEN-transcribe fusion (SPEC floor, GATE-2 decision 11).
///
/// Each segmentation span already belongs to exactly one speaker, so the span's
/// transcript is attributed to that speaker BY CONSTRUCTION — no fragile
/// word-timestamp reconciliation. Segments are processed in timeline order;
/// the local speaker slot (0..2) maps to a 1-based display label.
///
/// [onLine], if given, is called as each line is produced (progressive render,
/// R4). Empty-text spans are skipped from the result.
List<TranscriptLine> fuse(
  List<SpeakerSegment> segments,
  SpanTranscriber transcribe, {
  void Function(TranscriptLine line)? onLine,
}) {
  final List<SpeakerSegment> ordered = List<SpeakerSegment>.from(segments)
    ..sort((SpeakerSegment a, SpeakerSegment b) => a.start.compareTo(b.start));
  final List<TranscriptLine> lines = <TranscriptLine>[];
  for (final SpeakerSegment seg in ordered) {
    final String text = transcribe(seg).trim();
    if (text.isEmpty) continue;
    final TranscriptLine line = TranscriptLine(
      speaker: seg.speaker + 1, // 1-based for display
      start: seg.start,
      end: seg.end,
      text: text,
    );
    lines.add(line);
    onLine?.call(line);
  }
  return lines;
}

/// One speaker-attributed transcript line: a segmentation span's text tagged
/// with the speaker of that span (attribution is by construction — the span
/// already belongs to exactly one speaker; SPEC diarize-then-transcribe).
class TranscriptLine {
  const TranscriptLine({
    required this.speaker,
    required this.start,
    required this.end,
    required this.text,
  });

  /// 1-based speaker label for display ("Speaker 1", "Speaker 2", ...).
  final int speaker;

  /// Span start/end in seconds.
  final double start;
  final double end;

  /// Transcribed text for the span (may be empty if Whisper produced nothing).
  final String text;

  @override
  String toString() =>
      'TranscriptLine(Speaker $speaker, ${start.toStringAsFixed(2)}..'
      '${end.toStringAsFixed(2)}: "$text")';
}

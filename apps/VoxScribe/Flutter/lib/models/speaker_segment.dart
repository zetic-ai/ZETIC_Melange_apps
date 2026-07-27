/// A "who spoke when" turn produced by pyannote segmentation post-processing.
///
/// [speaker] is the LOCAL powerset speaker slot (0..2). In the FLOOR (a single
/// 10 s segmentation window) local slots ARE the global identities — no
/// stitching or clustering is performed (SPEC floor).
class SpeakerSegment {
  const SpeakerSegment({
    required this.start,
    required this.end,
    required this.speaker,
  });

  /// Segment start time in seconds (from the segmentation frame→time map).
  final double start;

  /// Segment end time in seconds.
  final double end;

  /// Local speaker slot, 0..2.
  final int speaker;

  double get duration => end - start;

  @override
  String toString() =>
      'SpeakerSegment(spk=$speaker, ${start.toStringAsFixed(3)}..'
      '${end.toStringAsFixed(3)})';
}

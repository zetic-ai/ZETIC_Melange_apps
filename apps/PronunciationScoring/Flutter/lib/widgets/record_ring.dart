import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme.dart';

/// The recording control: a mic button wrapped in a countdown ring that fills
/// over the fixed 5.11 s window. Tapping while idle starts a capture; tapping
/// while recording CANCELS (discards — a partial window is never scored).
class RecordRing extends StatelessWidget {
  const RecordRing({
    super.key,
    required this.progress,
    required this.recording,
    required this.scoring,
    required this.onStart,
    required this.onCancel,
  });

  /// 0..1 fill of the 5.11 s window.
  final double progress;
  final bool recording;
  final bool scoring;
  final VoidCallback onStart;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    const size = 132.0;
    return GestureDetector(
      onTap: scoring ? null : (recording ? onCancel : onStart),
      child: SizedBox(
        width: size,
        height: size,
        child: CustomPaint(
          painter: _RingPainter(
            progress: recording ? progress : (scoring ? 1 : 0),
            active: recording,
          ),
          child: Center(
            child: scoring
                ? const SizedBox(
                    width: 42,
                    height: 42,
                    child: CircularProgressIndicator(strokeWidth: 3),
                  )
                : Icon(
                    recording ? Icons.stop_rounded : Icons.mic_rounded,
                    color: Colors.white,
                    size: 52,
                  ),
          ),
        ),
      ),
    );
  }
}

class _RingPainter extends CustomPainter {
  _RingPainter({required this.progress, required this.active});

  final double progress;
  final bool active;

  @override
  void paint(Canvas canvas, Size size) {
    final center = size.center(Offset.zero);
    final radius = size.width / 2 - 8;

    final track = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 8
      ..color = Colors.white.withValues(alpha: 0.12);
    canvas.drawCircle(center, radius, track);

    // Filled inner disc.
    final disc = Paint()
      ..style = PaintingStyle.fill
      ..color = active
          ? SayColors.accent.withValues(alpha: 0.9)
          : SayColors.accent.withValues(alpha: 0.7);
    canvas.drawCircle(center, radius - 10, disc);

    if (progress > 0) {
      final arc = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 8
        ..strokeCap = StrokeCap.round
        ..color = SayColors.accentSoft;
      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        -math.pi / 2,
        2 * math.pi * progress.clamp(0.0, 1.0),
        false,
        arc,
      );
    }
  }

  @override
  bool shouldRepaint(_RingPainter old) =>
      old.progress != progress || old.active != active;
}

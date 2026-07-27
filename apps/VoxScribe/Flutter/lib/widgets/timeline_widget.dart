import 'package:flutter/material.dart';

import '../models/speaker_segment.dart';
import 'hud.dart';

/// "Who spoke when" timeline: one lane per local speaker, with colored bands
/// over [0, durationSec]. Maps directly onto Kardome's positioning and is cheap
/// since the segments are already computed.
class TimelineWidget extends StatelessWidget {
  const TimelineWidget({
    super.key,
    required this.segments,
    required this.durationSec,
  });

  final List<SpeakerSegment> segments;
  final double durationSec;

  @override
  Widget build(BuildContext context) {
    final int lanes = segments.isEmpty
        ? 1
        : (segments.map((SpeakerSegment s) => s.speaker).reduce(
                  (int a, int b) => a > b ? a : b,
                ) +
            1);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        const Padding(
          padding: EdgeInsets.only(left: 4, bottom: 6),
          child: Text('Who spoke when',
              style: TextStyle(
                  color: Colors.white54,
                  fontSize: 12,
                  fontWeight: FontWeight.w600)),
        ),
        SizedBox(
          height: lanes * 26.0 + 18,
          child: CustomPaint(
            size: Size.infinite,
            painter: _TimelinePainter(
              segments: segments,
              durationSec: durationSec <= 0 ? 1 : durationSec,
              lanes: lanes,
            ),
          ),
        ),
      ],
    );
  }
}

class _TimelinePainter extends CustomPainter {
  _TimelinePainter({
    required this.segments,
    required this.durationSec,
    required this.lanes,
  });

  final List<SpeakerSegment> segments;
  final double durationSec;
  final int lanes;

  @override
  void paint(Canvas canvas, Size size) {
    const double laneH = 20;
    const double laneGap = 6;
    final Paint bg = Paint()..color = Colors.white.withValues(alpha: 0.05);
    final double axisY = lanes * (laneH + laneGap);

    for (int lane = 0; lane < lanes; lane++) {
      final double y = lane * (laneH + laneGap);
      canvas.drawRRect(
        RRect.fromRectAndRadius(
            Rect.fromLTWH(0, y, size.width, laneH), const Radius.circular(4)),
        bg,
      );
    }

    for (final SpeakerSegment s in segments) {
      final double x1 = (s.start / durationSec) * size.width;
      final double x2 = (s.end / durationSec) * size.width;
      final double y = s.speaker * (laneH + laneGap);
      final Paint p = Paint()
        ..color = speakerColor(s.speaker + 1).withValues(alpha: 0.85);
      canvas.drawRRect(
        RRect.fromRectAndRadius(
          Rect.fromLTWH(x1, y, (x2 - x1).clamp(2.0, size.width), laneH),
          const Radius.circular(4),
        ),
        p,
      );
    }

    // time axis ticks (every ~2 s)
    final TextPainter tp = TextPainter(textDirection: TextDirection.ltr);
    final Paint tick = Paint()..color = Colors.white24;
    for (double sec = 0; sec <= durationSec; sec += 2) {
      final double x = (sec / durationSec) * size.width;
      canvas.drawLine(Offset(x, axisY), Offset(x, axisY + 4), tick);
      tp
        ..text = TextSpan(
            text: '${sec.toStringAsFixed(0)}s',
            style: const TextStyle(color: Colors.white30, fontSize: 9))
        ..layout();
      tp.paint(canvas, Offset(x + 2, axisY + 4));
    }
  }

  @override
  bool shouldRepaint(_TimelinePainter old) =>
      old.segments != segments || old.durationSec != durationSec;
}

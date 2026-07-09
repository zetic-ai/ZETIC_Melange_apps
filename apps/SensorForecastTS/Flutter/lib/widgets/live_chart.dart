import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/forecast.dart';

/// Immutable snapshot of everything the chart draws for one revision.
class ChartData {
  const ChartData({
    required this.revision,
    required this.history,
    required this.historyStartIndex,
    required this.forecast,
    required this.anomalyIndices,
    required this.threshold,
  });

  /// Monotonic counter; the painter repaints only when this changes.
  final int revision;

  /// Visible tail of the signal (oldest first). history[k] is global sample
  /// index historyStartIndex + k.
  final List<double> history;
  final int historyStartIndex;

  final Forecast? forecast;

  /// Global indices of flagged anomalies that fall inside the visible range.
  final List<int> anomalyIndices;

  final double threshold;
}

class LiveChart extends StatelessWidget {
  const LiveChart({super.key, required this.data});

  final ChartData data;

  @override
  Widget build(BuildContext context) {
    return RepaintBoundary(
      child: CustomPaint(
        painter: _ChartPainter(data),
        size: Size.infinite,
      ),
    );
  }
}

class _ChartPainter extends CustomPainter {
  _ChartPainter(this.data);

  final ChartData data;

  static const _bg = Color(0xFF0B1220);
  static const _grid = Color(0xFF1D2A40);
  static const _trace = Color(0xFF3FE0C5);
  static const _median = Color(0xFF6FA8FF);
  static const _bandOuter = Color(0x2E6FA8FF);
  static const _bandInner = Color(0x3D6FA8FF);
  static const _nowLine = Color(0xFF8899AA);
  static const _anomaly = Color(0xFFFF5470);

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = _bg);
    final hist = data.history;
    if (hist.length < 2) return;

    final f = data.forecast;
    final firstIdx = data.historyStartIndex;
    final lastIdx = firstIdx + hist.length - 1;

    // X domain: visible history plus the forecast horizon ahead of "now".
    final domainStart = firstIdx;
    final domainEnd = lastIdx + kHorizon;
    final xPer = size.width / (domainEnd - domainStart);
    double xOf(num globalIdx) => (globalIdx - domainStart) * xPer;

    // Y domain: history plus forecast band, padded.
    var lo = double.infinity, hi = double.negativeInfinity;
    for (final v in hist) {
      lo = math.min(lo, v);
      hi = math.max(hi, v);
    }
    if (f != null) {
      for (var t = 0; t < kHorizon; t++) {
        lo = math.min(lo, f.q10(t));
        hi = math.max(hi, f.q90(t));
      }
    }
    final pad = math.max((hi - lo) * 0.15, 1.0);
    lo -= pad;
    hi += pad;
    double yOf(double v) => size.height * (1 - (v - lo) / (hi - lo));

    // Grid.
    final gridPaint = Paint()
      ..color = _grid
      ..strokeWidth = 1;
    for (var g = 1; g < 5; g++) {
      final y = size.height * g / 5;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // Forecast fan (under the trace).
    if (f != null) {
      void band(double Function(int) top, double Function(int) bottom,
          Color color) {
        final path = Path()..moveTo(xOf(f.anchorIndex), yOf(top(0)));
        for (var t = 1; t < kHorizon; t++) {
          path.lineTo(xOf(f.anchorIndex + t), yOf(top(t)));
        }
        for (var t = kHorizon - 1; t >= 0; t--) {
          path.lineTo(xOf(f.anchorIndex + t), yOf(bottom(t)));
        }
        path.close();
        canvas.drawPath(path, Paint()..color = color);
      }

      band(f.q90, f.q10, _bandOuter);
      band(f.q70, f.q30, _bandInner);

      final medianPath = Path()..moveTo(xOf(f.anchorIndex), yOf(f.median(0)));
      for (var t = 1; t < kHorizon; t++) {
        medianPath.lineTo(xOf(f.anchorIndex + t), yOf(f.median(t)));
      }
      canvas.drawPath(
        medianPath,
        Paint()
          ..color = _median
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
    }

    // History trace.
    final tracePath = Path()..moveTo(xOf(firstIdx), yOf(hist[0]));
    for (var k = 1; k < hist.length; k++) {
      tracePath.lineTo(xOf(firstIdx + k), yOf(hist[k]));
    }
    canvas.drawPath(
      tracePath,
      Paint()
        ..color = _trace
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2,
    );

    // "Now" divider.
    final nowX = xOf(lastIdx);
    canvas.drawLine(
      Offset(nowX, 0),
      Offset(nowX, size.height),
      Paint()
        ..color = _nowLine
        ..strokeWidth = 1,
    );

    // Anomaly markers.
    final markerPaint = Paint()..color = _anomaly;
    final ringPaint = Paint()
      ..color = _anomaly
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    for (final gi in data.anomalyIndices) {
      final k = gi - firstIdx;
      if (k < 0 || k >= hist.length) continue;
      final c = Offset(xOf(gi), yOf(hist[k]));
      canvas.drawCircle(c, 3.5, markerPaint);
      canvas.drawCircle(c, 7.0, ringPaint);
    }
  }

  @override
  bool shouldRepaint(_ChartPainter oldDelegate) =>
      oldDelegate.data.revision != data.revision;
}

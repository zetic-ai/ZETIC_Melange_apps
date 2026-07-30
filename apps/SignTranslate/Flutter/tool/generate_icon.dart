// Generates the GlyphGo launcher icon (1024x1024 PNG) with pure dart:ui —
// no external assets or downloads. Motif: a wayfinding waypoint pin whose
// head carries stylized "text line" glyph bars (scene-text reading), over a
// night-travel navy field with a dashed route line — the app palette
// (theme.dart: navy 0xFF0B1020/0xFF161D33, teal 0xFF2DD4BF, indigo
// 0xFF8B9DF8).
//
// Run: flutter test tool/generate_icon.dart
// Then: dart run flutter_launcher_icons
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter_test/flutter_test.dart';

const _size = 1024.0;

void main() {
  test('generate assets/icon/app_icon.png (1024x1024)', () async {
    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(
        recorder, const ui.Rect.fromLTWH(0, 0, _size, _size));

    _drawBackground(canvas);
    _drawRoute(canvas);
    _drawCompassRing(canvas);
    _drawPin(canvas);

    final image =
        await recorder.endRecording().toImage(_size.toInt(), _size.toInt());
    final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
    final file = File('assets/icon/app_icon.png');
    file.parent.createSync(recursive: true);
    file.writeAsBytesSync(bytes!.buffer.asUint8List());

    // Sanity: PNG magic + IHDR 1024x1024.
    final out = file.readAsBytesSync();
    expect(out.sublist(0, 8),
        [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    expect(_beU32(out, 16), 1024); // IHDR width
    expect(_beU32(out, 20), 1024); // IHDR height
  });
}

int _beU32(Uint8List b, int o) =>
    (b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3];

void _drawBackground(ui.Canvas canvas) {
  const rect = ui.Rect.fromLTWH(0, 0, _size, _size);
  canvas.drawRect(
    rect,
    ui.Paint()
      ..shader = ui.Gradient.linear(
        const ui.Offset(0, 0),
        const ui.Offset(_size, _size),
        [const ui.Color(0xFF161D33), const ui.Color(0xFF0B1020)],
      ),
  );
  // Soft teal glow behind the pin head.
  canvas.drawCircle(
    const ui.Offset(512, 430),
    360,
    ui.Paint()
      ..shader = ui.Gradient.radial(
        const ui.Offset(512, 430),
        360,
        [const ui.Color(0x332DD4BF), const ui.Color(0x002DD4BF)],
      ),
  );
}

/// Dashed route line sweeping from bottom-left up toward the pin tail.
void _drawRoute(ui.Canvas canvas) {
  final path = ui.Path()
    ..moveTo(96, 960)
    ..cubicTo(300, 950, 430, 900, 512, 812);
  final paint = ui.Paint()
    ..style = ui.PaintingStyle.stroke
    ..strokeWidth = 26
    ..strokeCap = ui.StrokeCap.round
    ..color = const ui.Color(0xB38B9DF8);
  for (final metric in path.computeMetrics()) {
    const dash = 58.0, gap = 46.0;
    var d = 0.0;
    while (d < metric.length) {
      final end = math.min(d + dash, metric.length);
      canvas.drawPath(metric.extractPath(d, end), paint);
      d = end + gap;
    }
  }
  // Origin dot of the route.
  canvas.drawCircle(const ui.Offset(96, 960), 26,
      ui.Paint()..color = const ui.Color(0xFF8B9DF8));
}

/// Faint compass ring behind the pin (travel motif).
void _drawCompassRing(ui.Canvas canvas) {
  final ring = ui.Paint()
    ..style = ui.PaintingStyle.stroke
    ..strokeWidth = 8
    ..color = const ui.Color(0x2E8B9DF8);
  canvas.drawCircle(const ui.Offset(512, 430), 332, ring);
  // Cardinal ticks.
  final tick = ui.Paint()
    ..strokeWidth = 10
    ..strokeCap = ui.StrokeCap.round
    ..color = const ui.Color(0x4D8B9DF8);
  for (var i = 0; i < 4; i++) {
    final a = i * math.pi / 2 - math.pi / 2;
    final c = ui.Offset(512 + math.cos(a) * 332, 430 + math.sin(a) * 332);
    final o = ui.Offset(math.cos(a) * 30, math.sin(a) * 30);
    canvas.drawLine(c - o, c + o, tick);
  }
}

/// The waypoint pin: teal head + tail, navy glyph "text lines" inside.
void _drawPin(ui.Canvas canvas) {
  const cx = 512.0, cy = 430.0, r = 250.0, tipY = 812.0;

  // Tail: two arcs from the circle's lower tangents to the tip.
  final tail = ui.Path()
    ..moveTo(cx - r * 0.62, cy + r * 0.72)
    ..quadraticBezierTo(cx - r * 0.30, cy + r * 1.28, cx, tipY)
    ..quadraticBezierTo(cx + r * 0.30, cy + r * 1.28, cx + r * 0.62,
        cy + r * 0.72)
    ..close();

  final teal = ui.Paint()
    ..shader = ui.Gradient.linear(
      const ui.Offset(cx, cy - r),
      const ui.Offset(cx, tipY),
      [const ui.Color(0xFF2DD4BF), const ui.Color(0xFF14B8A6)],
    );
  canvas.drawPath(tail, teal);
  canvas.drawCircle(const ui.Offset(cx, cy), r, teal);

  // Subtle darker rim so the pin reads at 60x60.
  canvas.drawCircle(
    const ui.Offset(cx, cy),
    r,
    ui.Paint()
      ..style = ui.PaintingStyle.stroke
      ..strokeWidth = 14
      ..color = const ui.Color(0x330B1020),
  );

  // Glyph bars: three rounded "text lines" (the reading motif), navy.
  final bar = ui.Paint()..color = const ui.Color(0xFF0B1020);
  const bh = 52.0; // bar height
  const rr = ui.Radius.circular(bh / 2);
  canvas.drawRRect(
      ui.RRect.fromRectAndRadius(
          ui.Rect.fromCenter(
              center: const ui.Offset(cx, cy - 92),
              width: 268,
              height: bh),
          rr),
      bar);
  canvas.drawRRect(
      ui.RRect.fromRectAndRadius(
          ui.Rect.fromCenter(
              center: const ui.Offset(cx, cy), width: 196, height: bh),
          rr),
      bar);
  canvas.drawRRect(
      ui.RRect.fromRectAndRadius(
          ui.Rect.fromCenter(
              center: const ui.Offset(cx, cy + 92),
              width: 244,
              height: bh),
          rr),
      bar);
  // Cursor dot finishing the middle line (live-reading hint).
  canvas.drawCircle(const ui.Offset(cx + 128, cy), bh / 2, bar);
}

// Generates the OraLens launcher icon (1024x1024) — a stylized tooth inside a
// teal detection bounding box with a scan line, on a clinical dark ground.
// Run: dart run tool/gen_icon.dart  (writes assets/icon/app_icon.png)
import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

void main() {
  const int s = 1024;
  final img.Image im = img.Image(width: s, height: s);

  // Background: clinical dark.
  img.fill(im, color: img.ColorRgb8(0x0E, 0x1A, 0x24));

  final img.Color tooth = img.ColorRgb8(0xEA, 0xF2, 0xF5);
  final img.Color teal = img.ColorRgb8(0x2F, 0xD4, 0xC4);

  // --- Tooth ---------------------------------------------------------------
  // Crown: two merged lobes + body block.
  img.fillCircle(im, x: 420, y: 430, radius: 150, color: tooth);
  img.fillCircle(im, x: 604, y: 430, radius: 150, color: tooth);
  img.fillRect(im, x1: 300, y1: 420, x2: 724, y2: 560, color: tooth);
  img.fillCircle(im, x: 300, y: 470, radius: 90, color: tooth);
  img.fillCircle(im, x: 724, y: 470, radius: 90, color: tooth);

  // Two roots tapering downward.
  img.fillPolygon(im, vertices: <img.Point>[
    img.Point(300, 545),
    img.Point(478, 545),
    img.Point(392, 800),
  ], color: tooth);
  img.fillPolygon(im, vertices: <img.Point>[
    img.Point(546, 545),
    img.Point(724, 545),
    img.Point(632, 800),
  ], color: tooth);

  // Crown notch between the lobes (carve back to background).
  img.fillPolygon(im, vertices: <img.Point>[
    img.Point(486, 300),
    img.Point(538, 300),
    img.Point(512, 400),
  ], color: img.ColorRgb8(0x0E, 0x1A, 0x24));

  // --- Detection bounding box (teal) ---------------------------------------
  img.drawRect(im,
      x1: 235, y1: 250, x2: 789, y2: 812, color: teal, thickness: 16);
  // Corner ticks for a "detector" feel.
  const int t = 70;
  for (final List<int> c in <List<int>>[
    <int>[235, 250, 1, 1],
    <int>[789, 250, -1, 1],
    <int>[235, 812, 1, -1],
    <int>[789, 812, -1, -1],
  ]) {
    final int x = c[0], y = c[1], dx = c[2], dy = c[3];
    img.fillRect(im,
        x1: math.min(x, x + dx * t),
        y1: math.min(y, y + dy * 26),
        x2: math.max(x, x + dx * t),
        y2: math.max(y, y + dy * 26),
        color: teal);
    img.fillRect(im,
        x1: math.min(x, x + dx * 26),
        y1: math.min(y, y + dy * t),
        x2: math.max(x, x + dx * 26),
        y2: math.max(y, y + dy * t),
        color: teal);
  }

  // Scan line across the tooth.
  img.fillRect(im,
      x1: 235, y1: 520, x2: 789, y2: 532,
      color: img.ColorRgba8(0x2F, 0xD4, 0xC4, 150));

  final File out = File('assets/icon/app_icon.png');
  out.writeAsBytesSync(img.encodePng(im));
  stdout.writeln('wrote ${out.path} (${im.width}x${im.height})');
}

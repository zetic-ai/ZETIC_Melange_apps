// Generates assets/icon/app_icon.png (1024x1024) — a shelf/planogram glyph with
// bright "scan-green" detection boxes. Run: dart run tool/gen_icon.dart
import 'dart:io';
import 'package:image/image.dart' as img;

img.ColorRgb8 c(int r, int g, int b) => img.ColorRgb8(r, g, b);

void main() {
  const s = 1024;
  final im = img.Image(width: s, height: s, numChannels: 3);

  // Dark background with a subtle vertical gradient (retail-HUD navy).
  for (var y = 0; y < s; y++) {
    final t = y / s;
    final r = (11 + t * 10).round();
    final g = (15 + t * 18).round();
    final b = (20 + t * 26).round();
    img.fillRect(im, x1: 0, y1: y, x2: s - 1, y2: y, color: c(r, g, b));
  }

  final shelfCol = c(58, 74, 92); // shelf bars
  final accent = c(19, 226, 123); // scan green
  final prodA = c(90, 104, 122);
  final prodB = c(120, 132, 150);

  // Layout: 3 shelves with product facings; a couple wrapped in green boxes.
  const left = 150, right = 874, top = 210, bottom = 840;
  const shelves = 3;
  final shelfH = (bottom - top) ~/ shelves;

  for (var i = 0; i < shelves; i++) {
    final sy = top + i * shelfH;
    final baseY = sy + shelfH - 34;
    // shelf bar
    img.fillRect(im, x1: left, y1: baseY, x2: right, y2: baseY + 16, color: shelfCol);

    // product facings on the shelf, varied widths
    final widths = i == 0
        ? <int>[96, 70, 120, 84, 96]
        : i == 1
            ? <int>[80, 80, 80, 80, 80, 80]
            : <int>[140, 90, 140, 110];
    var x = left + 16;
    var k = 0;
    for (final w in widths) {
      if (x + w > right - 16) break;
      final h = 120 + (k.isEven ? 40 : 0) + i * 6;
      final py2 = baseY - 8;
      final py1 = py2 - h;
      img.fillRect(im, x1: x, y1: py1, x2: x + w, y2: py2,
          color: (k.isEven ? prodA : prodB));
      x += w + 22;
      k++;
    }
  }

  // Green "detected" boxes over a few facings (the money shot signal).
  void greenBox(int x1, int y1, int x2, int y2) {
    img.drawRect(im, x1: x1, y1: y1, x2: x2, y2: y2, color: accent, thickness: 10);
  }

  greenBox(166, 300, 262, 470);
  greenBox(360, 262, 480, 470);
  greenBox(196, 690, 336, 800);
  greenBox(536, 470, 616, 636);

  // Corner scan brackets to read as "detector".
  const m = 96, len = 120, th = 18;
  final br = accent;
  // top-left
  img.fillRect(im, x1: m, y1: m, x2: m + len, y2: m + th, color: br);
  img.fillRect(im, x1: m, y1: m, x2: m + th, y2: m + len, color: br);
  // top-right
  img.fillRect(im, x1: s - m - len, y1: m, x2: s - m, y2: m + th, color: br);
  img.fillRect(im, x1: s - m - th, y1: m, x2: s - m, y2: m + len, color: br);
  // bottom-left
  img.fillRect(im, x1: m, y1: s - m - th, x2: m + len, y2: s - m, color: br);
  img.fillRect(im, x1: m, y1: s - m - len, x2: m + th, y2: s - m, color: br);
  // bottom-right
  img.fillRect(im, x1: s - m - len, y1: s - m - th, x2: s - m, y2: s - m, color: br);
  img.fillRect(im, x1: s - m - th, y1: s - m - len, x2: s - m, y2: s - m, color: br);

  final out = File('assets/icon/app_icon.png');
  out.writeAsBytesSync(img.encodePng(im));
  stdout.writeln('wrote ${out.path} (${out.lengthSync()} bytes)');
}

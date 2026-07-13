// Generates assets/icon/app_icon.png (1024x1024): a document with text lines,
// two redaction bars, and a teal scanning lens — the RedactLens glyph.
// Run from Flutter/:  dart run tool/generate_icon.dart
import 'dart:io';

import 'package:image/image.dart' as img;

void main() {
  const size = 1024;
  final canvas = img.Image(width: size, height: size, numChannels: 4);

  final bgTop = img.ColorRgba8(16, 26, 34, 255); // deep navy
  final bgBottom = img.ColorRgba8(7, 12, 17, 255);
  final paper = img.ColorRgba8(236, 242, 247, 255);
  final paperEdge = img.ColorRgba8(203, 213, 225, 255);
  final line = img.ColorRgba8(148, 163, 184, 255);
  final bar = img.ColorRgba8(15, 23, 42, 255); // redaction bar
  final teal = img.ColorRgba8(45, 212, 191, 255);
  final tealDim = img.ColorRgba8(45, 212, 191, 90);

  // Vertical background gradient.
  for (var y = 0; y < size; y++) {
    final t = y / size;
    final c = img.ColorRgba8(
      (16 + (7 - 16) * t).round(),
      (26 + (12 - 26) * t).round(),
      (34 + (17 - 34) * t).round(),
      255,
    );
    img.drawLine(canvas, x1: 0, y1: y, x2: size - 1, y2: y, color: c);
  }
  // Referenced to keep the palette explicit even though the gradient covers it.
  img.drawPixel(canvas, 0, 0, bgTop);
  img.drawPixel(canvas, 0, size - 1, bgBottom);

  // Document sheet.
  const docX0 = 262, docY0 = 172, docX1 = 762, docY1 = 852;
  img.fillRect(canvas,
      x1: docX0 + 10,
      y1: docY0 + 14,
      x2: docX1 + 10,
      y2: docY1 + 14,
      color: img.ColorRgba8(0, 0, 0, 90),
      radius: 36); // drop shadow
  img.fillRect(canvas,
      x1: docX0, y1: docY0, x2: docX1, y2: docY1, color: paper, radius: 36);
  img.drawRect(canvas,
      x1: docX0, y1: docY0, x2: docX1, y2: docY1, color: paperEdge, radius: 36);

  // Text lines + redaction bars.
  const lx0 = docX0 + 56;
  const lx1 = docX1 - 56;
  const rowH = 34;
  const rows = [
    (y: docY0 + 90, redact: false, w: 1.0),
    (y: docY0 + 90 + 84, redact: true, w: 0.86),
    (y: docY0 + 90 + 168, redact: false, w: 0.94),
    (y: docY0 + 90 + 252, redact: true, w: 0.70),
    (y: docY0 + 90 + 336, redact: false, w: 1.0),
    (y: docY0 + 90 + 420, redact: false, w: 0.55),
  ];
  for (final row in rows) {
    final x1 = lx0 + ((lx1 - lx0) * row.w).round();
    if (row.redact) {
      img.fillRect(canvas,
          x1: lx0 - 8,
          y1: row.y - 8,
          x2: x1 + 8,
          y2: row.y + rowH + 8,
          color: bar,
          radius: 12);
      img.drawRect(canvas,
          x1: lx0 - 8,
          y1: row.y - 8,
          x2: x1 + 8,
          y2: row.y + rowH + 8,
          color: teal,
          radius: 12,
          thickness: 4);
    } else {
      img.fillRect(canvas,
          x1: lx0, y1: row.y, x2: x1, y2: row.y + rowH, color: line, radius: 10);
    }
  }

  // Teal lens over the lower-right of the document.
  const cx = 700, cy = 760, rOuter = 168, rInner = 128;
  img.fillCircle(canvas, x: cx, y: cy, radius: rOuter, color: teal);
  img.fillCircle(canvas, x: cx, y: cy, radius: rInner, color: tealDim);
  // Punch the lens interior back toward dark glass.
  img.fillCircle(
      canvas, x: cx, y: cy, radius: rInner, color: img.ColorRgba8(10, 20, 26, 235));
  // Scan line across the lens.
  img.fillRect(canvas,
      x1: cx - rInner + 14,
      y1: cy - 8,
      x2: cx + rInner - 14,
      y2: cy + 8,
      color: teal,
      radius: 8);
  // Lens handle.
  for (var i = 0; i < 90; i++) {
    img.fillCircle(canvas,
        x: cx + rOuter - 26 + i, y: cy + rOuter - 26 + i, radius: 34, color: teal);
  }

  File('assets/icon/app_icon.png').writeAsBytesSync(img.encodePng(canvas));
  // ignore: avoid_print
  print('wrote assets/icon/app_icon.png (${size}x$size)');
}

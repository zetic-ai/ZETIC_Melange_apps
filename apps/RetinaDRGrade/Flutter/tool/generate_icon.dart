// Generates the 1024x1024 launcher-icon source: a stylized fundus (retina) glyph
// wrapped in a 5-segment severity-gradient ring (green -> red) — the
// domain-identifying mark for GradeVue (a DR SEVERITY grader, distinct from the
// sibling FundusGate binary screener). Run from the Flutter/ dir:
//
//   dart run tool/generate_icon.dart
//
// Then regenerate platform icons with: dart run flutter_launcher_icons
import 'dart:io';
import 'dart:math' as math;

import 'package:image/image.dart' as img;

void main() {
  const size = 1024;
  final image = img.Image(width: size, height: size, numChannels: 3);

  // Dark clinical background.
  img.fill(image, color: img.ColorRgb8(0x0B, 0x14, 0x18));

  const cx = size / 2;
  const cy = size / 2;

  // 5-segment severity-gradient ring (grade 0 green -> grade 4 red), the visual
  // signature of a SEVERITY grader. Drawn as thick arcs of short line segments.
  final gradeColors = <img.ColorRgb8>[
    img.ColorRgb8(0x3D, 0xBE, 0x8B), // 0 No DR
    img.ColorRgb8(0x8F, 0xCF, 0x6A), // 1 Mild
    img.ColorRgb8(0xE8, 0xB2, 0x3A), // 2 Moderate
    img.ColorRgb8(0xE4, 0x72, 0x2E), // 3 Severe
    img.ColorRgb8(0xE4, 0x57, 0x2E), // 4 Proliferative
  ];
  const ringR = 470.0;
  const gap = 0.06; // radians of gap between segments
  for (var s = 0; s < 5; s++) {
    final start = -math.pi / 2 + s * (2 * math.pi / 5) + gap / 2;
    final end = start + (2 * math.pi / 5) - gap;
    final color = gradeColors[s];
    for (var a = start; a < end; a += 0.004) {
      final x = cx + math.cos(a) * ringR;
      final y = cy + math.sin(a) * ringR;
      img.fillCircle(
        image,
        x: x.toInt(),
        y: y.toInt(),
        radius: 22,
        color: color,
      );
    }
  }

  // Fundus body (reddish-orange retina).
  img.fillCircle(
    image,
    x: cx.toInt(),
    y: cy.toInt(),
    radius: 400,
    color: img.ColorRgb8(0xB5, 0x43, 0x2C),
  );

  // Optic disc (bright, offset from center).
  final discX = (cx + 145).toInt();
  final discY = (cy - 38).toInt();
  img.fillCircle(
    image,
    x: discX,
    y: discY,
    radius: 88,
    color: img.ColorRgb8(0xFF, 0xD2, 0x7A),
  );

  // Retinal vessels radiating from the optic disc.
  final vessel = img.ColorRgb8(0x7A, 0x1E, 0x14);
  final rand = math.Random(11);
  for (var i = 0; i < 7; i++) {
    final angle = (i / 7) * 2 * math.pi + 0.3;
    var x = discX.toDouble();
    var y = discY.toDouble();
    var a = angle;
    for (var seg = 0; seg < 9; seg++) {
      final len = 52.0 + rand.nextDouble() * 30;
      final nx = x + math.cos(a) * len;
      final ny = y + math.sin(a) * len;
      img.drawLine(
        image,
        x1: x.toInt(),
        y1: y.toInt(),
        x2: nx.toInt(),
        y2: ny.toInt(),
        color: vessel,
        thickness: (7 - seg ~/ 2).clamp(2, 7),
      );
      x = nx;
      y = ny;
      a += (rand.nextDouble() - 0.5) * 0.7;
    }
  }

  final outFile = File('assets/icon/app_icon.png');
  outFile.parent.createSync(recursive: true);
  outFile.writeAsBytesSync(img.encodePng(image));
  stdout.writeln('Wrote ${outFile.path}');
}

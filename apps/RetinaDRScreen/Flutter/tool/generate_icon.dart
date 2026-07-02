// Generates the 1024x1024 launcher-icon source: a stylized fundus (retina)
// glyph — the domain-identifying mark for FundusGate. Run from the Flutter/ dir:
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

  // Teal framing ring (the "gate").
  for (var t = 0; t < 26; t++) {
    img.drawCircle(
      image,
      x: cx.toInt(),
      y: cy.toInt(),
      radius: 470 - t,
      color: img.ColorRgb8(0x1F, 0xB6, 0xB0),
    );
  }

  // Fundus body (reddish-orange retina).
  img.fillCircle(
    image,
    x: cx.toInt(),
    y: cy.toInt(),
    radius: 420,
    color: img.ColorRgb8(0xB5, 0x43, 0x2C),
  );

  // Optic disc (bright, offset from center).
  final discX = (cx + 150).toInt();
  final discY = (cy - 40).toInt();
  img.fillCircle(
    image,
    x: discX,
    y: discY,
    radius: 92,
    color: img.ColorRgb8(0xFF, 0xD2, 0x7A),
  );

  // Retinal vessels radiating from the optic disc.
  final vessel = img.ColorRgb8(0x7A, 0x1E, 0x14);
  final rand = math.Random(7);
  for (var i = 0; i < 7; i++) {
    final angle = (i / 7) * 2 * math.pi + 0.3;
    var x = discX.toDouble();
    var y = discY.toDouble();
    var a = angle;
    for (var seg = 0; seg < 9; seg++) {
      final len = 55.0 + rand.nextDouble() * 30;
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

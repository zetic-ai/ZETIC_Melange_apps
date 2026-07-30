import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/services/detector_preprocessor.dart';

BgrFrame solidFrame(int w, int h, int b, int g, int r) {
  final bytes = Uint8List(w * h * 3);
  for (var i = 0; i < w * h; i++) {
    bytes[i * 3] = b;
    bytes[i * 3 + 1] = g;
    bytes[i * 3 + 2] = r;
  }
  return BgrFrame(w, h, bytes);
}

double normalized(int px, int channel) =>
    (px / 255.0 - kDetMean[channel]) / kDetStd[channel];

void main() {
  const size = kDetInputSize;
  const area = size * size;

  test('letterbox size is 736 — NOT 640', () {
    expect(kDetInputSize, 736);
  });

  group('letterbox-736 geometry', () {
    test('1280x720 frame: scale, centered pads computed correctly', () {
      final geo = computeLetterboxGeometry(1280, 720);
      expect(geo.scale, closeTo(736 / 1280, 1e-9));
      // newH = round(720 * 736/1280) = 414; padY = (736-414)/2 = 161.
      expect(geo.padX, 0);
      expect(geo.padY, 161);
    });

    test('portrait 720x1280 frame pads horizontally', () {
      final geo = computeLetterboxGeometry(720, 1280);
      expect(geo.scale, closeTo(736 / 1280, 1e-9));
      expect(geo.padY, 0);
      expect(geo.padX, (736 - (720 * 736 / 1280).round()) ~/ 2);
    });
  });

  group('fused normalize (ImageNet mean/std in BGR channel order 0,1,2)', () {
    test('a known BGR pixel produces hand-computed per-channel values', () {
      // B=52, G=140, R=230 everywhere.
      final out = Float32List(3 * area);
      final geo = letterboxDetectorInput(solidFrame(736, 736, 52, 140, 230), out);
      expect(geo.padX, 0);
      expect(geo.padY, 0);

      final center = (size ~/ 2) * size + size ~/ 2;
      expect(out[center], closeTo(normalized(52, 0), 1e-5)); // plane 0 = B
      expect(out[area + center], closeTo(normalized(140, 1), 1e-5)); // G
      expect(out[2 * area + center], closeTo(normalized(230, 2), 1e-5)); // R
    });

    test('BGR assertion: a pure-BLUE frame is hottest in plane 0 after '
        'de-normalization (an RGB swap would put it in plane 2)', () {
      final out = Float32List(3 * area);
      letterboxDetectorInput(solidFrame(736, 736, 255, 0, 0), out);
      final center = (size ~/ 2) * size + size ~/ 2;
      // Recover pixel values: px = (v*std + mean)*255.
      final b = (out[center] * kDetStd[0] + kDetMean[0]) * 255;
      final g = (out[area + center] * kDetStd[1] + kDetMean[1]) * 255;
      final r = (out[2 * area + center] * kDetStd[2] + kDetMean[2]) * 255;
      expect(b, closeTo(255, 0.5));
      expect(g, closeTo(0, 0.5));
      expect(r, closeTo(0, 0.5));
    });
  });

  group('NCHW planar layout', () {
    test('channel planes are contiguous with stride 736*736', () {
      final out = Float32List(3 * area);
      letterboxDetectorInput(solidFrame(736, 736, 255, 128, 0), out);
      // Every element of plane 0 equals normalized(255,0); plane 2 equals
      // normalized(0,2) — proving planar (not interleaved) layout.
      expect(out[123], closeTo(normalized(255, 0), 1e-5));
      expect(out[area + 123], closeTo(normalized(128, 1), 1e-5));
      expect(out[2 * area + 123], closeTo(normalized(0, 2), 1e-5));
    });
  });

  group('padding', () {
    test('pad region is 0.0 in tensor space; content region is not', () {
      final out = Float32List(3 * area);
      // 1280x720 -> content rows 161..574, pad rows above/below.
      final geo = letterboxDetectorInput(solidFrame(1280, 720, 90, 90, 90), out);
      expect(geo.padY, 161);
      expect(out[10 * size + 100], 0.0); // top pad
      expect(out[(size - 10) * size + 100], 0.0); // bottom pad
      expect(out[368 * size + 368], isNot(0.0)); // center content
    });
  });

  test('pre-allocated buffer: second frame fully overwrites the first', () {
    final out = Float32List(3 * area);
    letterboxDetectorInput(solidFrame(736, 736, 255, 255, 255), out);
    final before = out[area + 500];
    letterboxDetectorInput(solidFrame(736, 736, 0, 0, 0), out);
    expect(out[area + 500], isNot(before));
    expect(out[area + 500], closeTo(normalized(0, 1), 1e-5));
  });
}

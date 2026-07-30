import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/services/quad_deskew.dart';
import 'package:signtranslate/services/rec_preprocessor.dart';

BgrCrop solidCrop(int w, int h, int b, int g, int r) {
  final bytes = Uint8List(w * h * 3);
  for (var i = 0; i < w * h; i++) {
    bytes[i * 3] = b;
    bytes[i * 3 + 1] = g;
    bytes[i * 3 + 2] = r;
  }
  return BgrCrop(w, h, bytes);
}

void main() {
  const area = kRecHeight * kRecWidth;

  test('shape constants are SPEC-exact: 48x320, [-1,1] norm', () {
    expect(kRecHeight, 48);
    expect(kRecWidth, 320);
  });

  group('pad-not-stretch (SPEC: right-pad with zeros to width 320)', () {
    test('narrow crop: content resized to width round(48*w/h), no stretch',
        () {
      final out = Float32List(3 * area);
      // 100x48 source -> ratio 100/48 -> resizedW = round(48*100/48) = 100.
      final resizedW = recognizerPreprocess(solidCrop(100, 48, 255, 255, 255), out);
      expect(resizedW, 100);

      // Content region is white -> normalized +1.0 in every channel.
      for (var c = 0; c < 3; c++) {
        expect(out[c * area + 0], closeTo(1.0, 1e-6));
        expect(out[c * area + resizedW - 1], closeTo(1.0, 1e-6));
      }
    });

    test('pad region is EXACTLY 0.0 in NORMALIZED tensor space (ruling #1: '
        'PaddleOCR pads post-normalization; black pixels would be -1.0)', () {
      final out = Float32List(3 * area);
      final resizedW = recognizerPreprocess(solidCrop(96, 48, 200, 200, 200), out);
      expect(resizedW, 96);

      for (var c = 0; c < 3; c++) {
        for (var y = 0; y < kRecHeight; y++) {
          for (var x = resizedW; x < kRecWidth; x++) {
            final v = out[c * area + y * kRecWidth + x];
            expect(v, 0.0,
                reason: 'pad at c=$c y=$y x=$x must be 0.0, got $v');
            expect(v, isNot(-1.0));
          }
        }
      }
    });

    test('over-wide crop is capped at 320 — never stretched taller', () {
      final out = Float32List(3 * area);
      // 960x48 -> round(48*960/48)=960 -> capped at 320.
      final resizedW = recognizerPreprocess(solidCrop(960, 48, 10, 10, 10), out);
      expect(resizedW, kRecWidth);
      // No pad at all: last column is content.
      expect(out[kRecWidth - 1], isNot(0.0));
    });

    test('aspect is preserved: a half-width dark bar stays half-width', () {
      // 200x50 crop, left 100 columns black, right 100 columns white.
      final bytes = Uint8List(200 * 50 * 3);
      for (var y = 0; y < 50; y++) {
        for (var x = 100; x < 200; x++) {
          final i = (y * 200 + x) * 3;
          bytes[i] = 255;
          bytes[i + 1] = 255;
          bytes[i + 2] = 255;
        }
      }
      final out = Float32List(3 * area);
      // resizedW = round(48*200/50) = 192.
      final resizedW = recognizerPreprocess(BgrCrop(200, 50, bytes), out);
      expect(resizedW, 192);
      // Mid-height row: column 40 (in the black half, <96) ~ -1, column 150
      // (white half, >96) ~ +1 — the boundary sits at 96 = resizedW/2.
      final row = 24 * kRecWidth;
      expect(out[row + 40], closeTo(-1.0, 1e-2));
      expect(out[row + 150], closeTo(1.0, 1e-2));
    });
  });

  group('[-1,1] normalization: (px/255 - 0.5)/0.5', () {
    test('endpoints and midpoint', () {
      final out = Float32List(3 * area);
      recognizerPreprocess(solidCrop(320, 48, 0, 128, 255), out);
      expect(out[0], closeTo(-1.0, 1e-6)); // B=0 -> -1
      expect(out[area], closeTo(128 / 255 * 2 - 1, 1e-6)); // G=128 -> ~0.004
      expect(out[2 * area], closeTo(1.0, 1e-6)); // R=255 -> +1
    });
  });

  group('BGR channel order (no RGB swap anywhere)', () {
    test('pure BLUE crop lands hot in channel plane 0, cold in plane 2', () {
      final out = Float32List(3 * area);
      recognizerPreprocess(solidCrop(320, 48, 255, 0, 0), out);
      expect(out[0], closeTo(1.0, 1e-6)); // plane 0 = B
      expect(out[area], closeTo(-1.0, 1e-6)); // plane 1 = G
      expect(out[2 * area], closeTo(-1.0, 1e-6)); // plane 2 = R
    });

    test('pure RED crop lands hot in channel plane 2', () {
      final out = Float32List(3 * area);
      recognizerPreprocess(solidCrop(320, 48, 0, 0, 255), out);
      expect(out[0], closeTo(-1.0, 1e-6));
      expect(out[2 * area], closeTo(1.0, 1e-6));
    });
  });

  test('buffer is written in place and reused across calls', () {
    final out = Float32List(3 * area);
    recognizerPreprocess(solidCrop(320, 48, 255, 255, 255), out);
    expect(out[0], closeTo(1.0, 1e-6));
    recognizerPreprocess(solidCrop(160, 48, 0, 0, 0), out);
    expect(out[0], closeTo(-1.0, 1e-6)); // overwritten by second call
    expect(out[200], 0.0); // and the pad region re-zeroed (x=200 >= 160)
  });
}

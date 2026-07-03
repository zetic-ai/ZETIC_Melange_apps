import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/services/frame_data.dart';
import 'package:livedocredact/services/recognizer_preprocessor.dart';

/// Builds an upright BGRA test frame whose pixel (x, y) is given by [colorAt]
/// returning (b, g, r).
UprightFrame makeFrame(int w, int h, (int, int, int) Function(int x, int y) colorAt) {
  final bgra = Uint8List(w * h * 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final (b, g, r) = colorAt(x, y);
      final i = (y * w + x) * 4;
      bgra[i] = b;
      bgra[i + 1] = g;
      bgra[i + 2] = r;
      bgra[i + 3] = 255;
    }
  }
  return UprightFrame(FrameData.bgra8888(
    width: w,
    height: h,
    bgra: bgra,
    bgraRowStride: w * 4,
  ));
}

double at(Float32List t, int channel, int y, int x) =>
    t[channel * kRecHeight * kRecWidth + y * kRecWidth + x];

void main() {
  test('fixed-width pad-not-stretch: narrow crop -> H=48, right pad = 0.0', () {
    // Solid white 100x50 quad -> content width round(48*100/50) = 96.
    final frame = makeFrame(200, 100, (x, y) => (255, 255, 255));
    final quad = [
      const Offset(0, 0),
      const Offset(100, 0),
      const Offset(100, 50),
      const Offset(0, 50),
    ];
    expect(recognizerContentWidth(quad), 96);

    final t = preprocessRecognizerCrop(frame, quad);
    expect(t.length, kRecTensorLength);
    for (var c = 0; c < 3; c++) {
      // Content: white -> +1.0.
      expect(at(t, c, 0, 0), closeTo(1.0, 1e-6));
      expect(at(t, c, 47, 95), closeTo(1.0, 1e-6));
      // Pad: 0.0 in NORMALIZED space (PaddleOCR padding_im zeros — NOT the
      // -1.0 a pixel-space zero-pad would produce).
      expect(at(t, c, 0, 96), 0.0);
      expect(at(t, c, 24, 200), 0.0);
      expect(at(t, c, 47, 319), 0.0);
    }
  });

  test('content is aspect-resized, not stretched to 320', () {
    // Left half black / right half white inside a 100x50 quad: the black->
    // white boundary must sit at ~contentWidth/2 (48), NOT at 160 (which is
    // where a stretch-to-320 would put it).
    final frame = makeFrame(200, 100, (x, y) => x < 50 ? (0, 0, 0) : (255, 255, 255));
    final quad = [
      const Offset(0, 0),
      const Offset(100, 0),
      const Offset(100, 50),
      const Offset(0, 50),
    ];
    final t = preprocessRecognizerCrop(frame, quad);
    expect(at(t, 0, 24, 10), closeTo(-1.0, 1e-6), reason: 'left = black');
    expect(at(t, 0, 24, 90), closeTo(1.0, 1e-6), reason: 'right = white');
    expect(at(t, 0, 24, 46), closeTo(-1.0, 1e-6),
        reason: 'boundary at ~48, so 46 is still black');
    expect(at(t, 0, 24, 50), closeTo(1.0, 1e-6),
        reason: 'boundary at ~48, so 50 is already white');
    expect(at(t, 0, 24, 160), 0.0,
        reason: 'a stretched crop would still have content at 160; the '
            'aspect-preserving pipeline has pad there');
  });

  test('crops wider than 320 are downscaled to exactly 320', () {
    // 1000x50 quad -> natural width 960 -> capped/downscaled to 320.
    final frame =
        makeFrame(1200, 100, (x, y) => x < 500 ? (0, 0, 0) : (255, 255, 255));
    final quad = [
      const Offset(0, 0),
      const Offset(1000, 0),
      const Offset(1000, 50),
      const Offset(0, 50),
    ];
    expect(recognizerContentWidth(quad), kRecWidth);
    final t = preprocessRecognizerCrop(frame, quad);
    // Content reaches the last column (no pad)...
    expect(at(t, 0, 24, 319), closeTo(1.0, 1e-6));
    // ...and the black/white boundary lands at half the CONTENT (x=500/1000
    // -> column ~160).
    expect(at(t, 0, 24, 150), closeTo(-1.0, 1e-6));
    expect(at(t, 0, 24, 170), closeTo(1.0, 1e-6));
  });

  test('normalization is (pixel/255 - 0.5)/0.5 -> [-1, 1]', () {
    final frame = makeFrame(64, 64, (x, y) => (0, 128, 255));
    final quad = [
      const Offset(0, 0),
      const Offset(48, 0),
      const Offset(48, 48),
      const Offset(0, 48),
    ];
    final t = preprocessRecognizerCrop(frame, quad);
    expect(at(t, 0, 10, 10), closeTo(-1.0, 1e-6)); // 0 -> -1
    expect(at(t, 1, 10, 10), closeTo(128 / 127.5 - 1.0, 1e-6)); // ~0.004
    expect(at(t, 2, 10, 10), closeTo(1.0, 1e-6)); // 255 -> +1
  });

  test('channel order is BGR: pure blue lands in channel 0', () {
    final frame = makeFrame(64, 64, (x, y) => (255, 0, 0)); // pure blue
    final quad = [
      const Offset(0, 0),
      const Offset(48, 0),
      const Offset(48, 48),
      const Offset(0, 48),
    ];
    final t = preprocessRecognizerCrop(frame, quad);
    expect(at(t, 0, 5, 5), closeTo(1.0, 1e-6), reason: 'B in channel 0');
    expect(at(t, 1, 5, 5), closeTo(-1.0, 1e-6));
    expect(at(t, 2, 5, 5), closeTo(-1.0, 1e-6),
        reason: 'a silent BGR->RGB swap would put +1.0 here');
  });

  test('per-crop deskew: a sideways (90°) quad is warped upright', () {
    // Physical strip x in [40,60], y in [20,180]; text runs bottom-to-top.
    // Reading order: quad tl at (40,180), tr at (40,20) — row 0 runs tl->tr.
    // The frame is black below y=100 and white above, so the UPRIGHT crop
    // must be black on its left half and white on its right half.
    final frame =
        makeFrame(100, 200, (x, y) => y >= 100 ? (0, 0, 0) : (255, 255, 255));
    final quad = [
      const Offset(40, 180), // tl
      const Offset(40, 20), // tr
      const Offset(60, 20), // br
      const Offset(60, 180), // bl
    ];
    // Natural width 48*160/20 = 384 -> capped at 320.
    final t = preprocessRecognizerCrop(frame, quad);
    expect(at(t, 0, 24, 20), closeTo(-1.0, 1e-6),
        reason: 'left of upright crop = bottom of strip = black');
    expect(at(t, 0, 24, 300), closeTo(1.0, 1e-6),
        reason: 'right of upright crop = top of strip = white');
    // Fed WITHOUT deskew (axis-aligned bbox instead of the quad) the pattern
    // would be horizontal bands instead — rows, not columns, would differ.
    expect(at(t, 0, 5, 20), closeTo(at(t, 0, 40, 20), 1e-6),
        reason: 'columns are uniform vertically after a correct deskew');
  });

  test('degenerate quads produce a silent all-zero tensor, not a crash', () {
    final frame = makeFrame(32, 32, (x, y) => (255, 255, 255));
    final quad = [
      const Offset(5, 5),
      const Offset(5.2, 5),
      const Offset(5.2, 5.1),
      const Offset(5, 5.1),
    ];
    final t = preprocessRecognizerCrop(frame, quad);
    expect(t.every((v) => v == 0.0), isTrue);
  });
}

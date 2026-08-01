import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livedocredact/services/detector_preprocessor.dart';
import 'package:livedocredact/services/frame_data.dart';

UprightFrame makeBgraFrame(
  int w,
  int h,
  (int, int, int) Function(int x, int y) colorAt, {
  int rotationDegrees = 0,
}) {
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
    rotationDegrees: rotationDegrees,
  ));
}

const int kArea = kDetInputSize * kDetInputSize;

double at(Float32List t, int channel, int y, int x) =>
    t[channel * kArea + y * kDetInputSize + x];

double normed(int pixel, int channel) =>
    (pixel / 255.0 - kDetMean[channel]) / kDetStd[channel];

void main() {
  test('letterbox bookkeeping: 1280x720 -> scale 0.5, pad (0, 140)', () {
    final frame = makeBgraFrame(1280, 720, (x, y) => (0, 0, 0));
    final result = preprocessDetectorFrame(frame);
    final g = result.geometry;
    expect(g.scale, closeTo(0.5, 1e-9));
    expect(g.padX, 0);
    expect(g.padY, 140);
    expect(g.srcWidth, 1280);
    expect(g.srcHeight, 720);
  });

  test(
      'ImageNet mean/std applied INDEX-WISE to BGR '
      '(channel 0 = B with mean 0.485)', () {
    // Pure blue: BGRA bytes (255, 0, 0, 255).
    final frame = makeBgraFrame(1280, 720, (x, y) => (255, 0, 0));
    final t = preprocessDetectorFrame(frame).input;
    // Content pixel (letterbox content spans y 140..499).
    expect(at(t, 0, 320, 320), closeTo(normed(255, 0), 1e-5),
        reason: 'B=255 normalized with mean 0.485/std 0.229');
    expect(at(t, 1, 320, 320), closeTo(normed(0, 1), 1e-5));
    expect(at(t, 2, 320, 320), closeTo(normed(0, 2), 1e-5),
        reason: 'a silent BGR->RGB swap would put the 255 here instead');
    // The swapped value is measurably different — the assert above can
    // actually catch the swap.
    expect((normed(255, 0) - normed(255, 2)).abs(), greaterThan(0.1));
  });

  test('letterbox padding is normalized black per channel', () {
    final frame = makeBgraFrame(1280, 720, (x, y) => (200, 200, 200));
    final t = preprocessDetectorFrame(frame).input;
    for (var c = 0; c < 3; c++) {
      expect(at(t, c, 0, 0), closeTo(normed(0, c), 1e-5),
          reason: 'pad rows (y<140) are normalized black');
      expect(at(t, c, 639, 639), closeTo(normed(0, c), 1e-5));
      expect(at(t, c, 140, 0), closeTo(normed(200, c), 1e-5),
          reason: 'first content row is real pixels');
    }
  });

  test('NCHW layout: one distinctive pixel lands at the right flat offsets',
      () {
    // White pixel at source (10, 10) in an otherwise black 1280x720 frame.
    final frame = makeBgraFrame(
        1280, 720, (x, y) => (x == 10 && y == 10) ? (255, 255, 255) : (0, 0, 0));
    final t = preprocessDetectorFrame(frame).input;
    // scale 0.5, padY 140: source (10,10) -> model (5, 145).
    for (var c = 0; c < 3; c++) {
      expect(at(t, c, 145, 5), closeTo(normed(255, c), 1e-5),
          reason: 'channel $c plane, row-major inside the plane');
      expect(at(t, c, 145, 6), closeTo(normed(0, c), 1e-5));
      expect(at(t, c, 146, 5), closeTo(normed(0, c), 1e-5));
    }
  });

  test('YUV420 (Android) decodes to BGR through the same path', () {
    // Neutral gray: Y=128, U=V=128 -> B=G=R=128 exactly (BT.601 offsets 0).
    const w = 64, h = 64;
    final frame = UprightFrame(FrameData.yuv420(
      width: w,
      height: h,
      yPlane: Uint8List(w * h)..fillRange(0, w * h, 128),
      uPlane: Uint8List(w * h ~/ 4)..fillRange(0, w * h ~/ 4, 128),
      vPlane: Uint8List(w * h ~/ 4)..fillRange(0, w * h ~/ 4, 128),
      yRowStride: w,
      uvRowStride: w ~/ 2,
      uvPixelStride: 1,
    ));
    final t = preprocessDetectorFrame(frame).input;
    for (var c = 0; c < 3; c++) {
      expect(at(t, c, 320, 320), closeTo(normed(128, c), 1e-5));
    }
  });

  test('YUV420 chroma indexing: U shifts blue, not red', () {
    const w = 64, h = 64;
    // Left half U=228 (blue-heavy), right half U=28 (blue-poor); V neutral.
    final u = Uint8List(w * h ~/ 4);
    for (var y = 0; y < h ~/ 2; y++) {
      for (var x = 0; x < w ~/ 2; x++) {
        u[y * (w ~/ 2) + x] = x < w ~/ 4 ? 228 : 28;
      }
    }
    final frame = UprightFrame(FrameData.yuv420(
      width: w,
      height: h,
      yPlane: Uint8List(w * h)..fillRange(0, w * h, 128),
      uPlane: u,
      vPlane: Uint8List(w * h ~/ 4)..fillRange(0, w * h ~/ 4, 128),
      yRowStride: w,
      uvRowStride: w ~/ 2,
      uvPixelStride: 1,
    ));
    final t = preprocessDetectorFrame(frame).input;
    // Content occupies the full 640 (square source): left quarter vs right.
    final bLeft = at(t, 0, 320, 100);
    final bRight = at(t, 0, 320, 540);
    expect(bLeft, greaterThan(bRight),
        reason: 'high U must raise the BLUE channel (channel 0)');
    final rLeft = at(t, 2, 320, 100);
    final rRight = at(t, 2, 320, 540);
    expect((rLeft - rRight).abs(), lessThan(0.05),
        reason: 'U must not move the red channel');
  });

  group('orientation transform round-trip (PyroGuard lesson: measure, never assume)',
      () {
    test('rotation 0 is a pure passthrough', () {
      final frame = makeBgraFrame(
          100, 50, (x, y) => (x == 5 && y == 7) ? (255, 255, 255) : (0, 0, 0));
      expect(frame.width, 100);
      expect(frame.height, 50);
      expect(frame.sampleBgrPacked(5, 7), 0xFFFFFF);
      expect(frame.sampleBgrPacked(6, 7), 0x000000);
    });

    test('rotation 90 maps a known raw pixel to the expected upright spot',
        () {
      // Raw 100x50 buffer, white at raw (5, 7), declared as needing a 90°
      // clockwise rotation -> upright is 50x100 and the pixel must appear at
      // upright (x, y) with rawX = y, rawY = (uprightWidth-1) - x:
      // y = 5, x = 49 - 7 = 42.
      final frame = makeBgraFrame(
          100, 50, (x, y) => (x == 5 && y == 7) ? (255, 255, 255) : (0, 0, 0),
          rotationDegrees: 90);
      expect(frame.width, 50, reason: 'upright dims swap under 90°');
      expect(frame.height, 100);
      expect(frame.sampleBgrPacked(42, 5), 0xFFFFFF);
      expect(frame.sampleBgrPacked(41, 5), 0x000000);
      expect(frame.sampleBgrPacked(42, 6), 0x000000);
    });

    test('rotation 180 maps to the opposite corner', () {
      final frame = makeBgraFrame(
          100, 50, (x, y) => (x == 0 && y == 0) ? (255, 255, 255) : (0, 0, 0),
          rotationDegrees: 180);
      expect(frame.sampleBgrPacked(99, 49), 0xFFFFFF);
      expect(frame.sampleBgrPacked(0, 0), 0x000000);
    });
  });
}

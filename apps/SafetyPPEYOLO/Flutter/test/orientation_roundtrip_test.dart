import 'dart:typed_data';
import 'dart:ui' show Rect, Size;

import 'package:flutter_test/flutter_test.dart';
import 'package:siteguard/services/preprocessor.dart';
import 'package:siteguard/widgets/overlay_geometry.dart';

/// Builds a BGRA frame of [w]x[h] filled with black except a bright-red 3x3
/// block centered at ([px],[py]) — 3x3 so the bilinear taps see full red
/// (value 1.0) and the marker cleanly out-scores the 0.5 letterbox padding.
FrameData bgraFrameWithMarker(int w, int h, int px, int py,
    {int rotationDegrees = 0}) {
  final bytes = Uint8List(w * h * 4);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      final int idx = ((py + dy) * w + (px + dx)) * 4;
      bytes[idx] = 0; // B
      bytes[idx + 1] = 0; // G
      bytes[idx + 2] = 255; // R
      bytes[idx + 3] = 255; // A
    }
  }
  return FrameData.bgra8888(
    width: w,
    height: h,
    bgra: bytes,
    bgraRowStride: w * 4,
    rotationDegrees: rotationDegrees,
  );
}

/// Finds the red-channel argmax in the NCHW tensor, returning (x, y).
(int, int) redArgmax(Float32List input) {
  const int size = kInputSize;
  var best = -1.0;
  var bestIdx = 0;
  for (var i = 0; i < size * size; i++) {
    if (input[i] > best) {
      best = input[i];
      bestIdx = i;
    }
  }
  return (bestIdx % size, bestIdx ~/ size);
}

void main() {
  group('orientation: buffer -> model transform', () {
    test(
        'iOS upright BGRA buffer (rotation 0): marker maps straight through '
        '(no spurious rotation — the PyroGuard bug)', () {
      // 720x1280 portrait buffer, as PyroGuard measured on-device (upright).
      const w = 720, h = 1280;
      // Marker at upper-left quadrant (180, 320).
      final frame = bgraFrameWithMarker(w, h, 180, 320);
      final pre = Preprocessor().run(frame);

      expect(pre.srcWidth, w);
      expect(pre.srcHeight, h);
      // scale = 640/1280 = 0.5; padX = (640-360)/2 = 140.
      expect(pre.scale, closeTo(0.5, 1e-9));
      expect(pre.padX, 140);
      expect(pre.padY, 0);

      final (mx, my) = redArgmax(pre.input);
      // Expected: 180*0.5 + 140 = 230, 320*0.5 = 160 (+-2 for bilinear).
      expect(mx, inInclusiveRange(228, 232));
      expect(my, inInclusiveRange(158, 162));
    });

    test('Android sensor-90 buffer: marker lands where the upright scene has it',
        () {
      // Landscape 1280x720 sensor buffer; rotating 90 CW makes it upright
      // 720x1280. A pixel at raw (x=1000, y=100) is, in the upright frame, at
      // (ux, uy) where rawX = uy => uy = 1000, rawY = (srcW-1)-ux = 100 =>
      // ux = 619.
      const rawW = 1280, rawH = 720;
      final frame =
          bgraFrameWithMarker(rawW, rawH, 1000, 100, rotationDegrees: 90);
      final pre = Preprocessor().run(frame);

      expect(pre.srcWidth, 720);
      expect(pre.srcHeight, 1280);

      final (mx, my) = redArgmax(pre.input);
      // upright (619, 1000) -> scale 0.5, padX 140: (449.5, 500) +-2.
      expect(mx, inInclusiveRange(447, 452));
      expect(my, inInclusiveRange(498, 502));
    });
  });

  group('orientation: overlay cover-fit mapping round-trips', () {
    test('portrait phone, landscape previewSize, sensor 90', () {
      // previewSize as reported (sensor orientation): 1280x720. Sensor 90 =>
      // upright content is 720x1280 => aspect 0.5625.
      final aspect = uprightContentAspect(
        previewWidth: 1280,
        previewHeight: 720,
        sensorOrientation: 90,
      );
      expect(aspect, closeTo(720 / 1280, 1e-9));

      const widget = Size(390, 844); // iPhone-ish portrait widget space
      // 0.8 x 0.1 normalized = 576 x 128 CONTENT pixels — genuinely wide.
      const box = Rect.fromLTRB(0.10, 0.30, 0.90, 0.40);

      final mapped = mapRectCover(box, aspect, widget);
      final back = unmapRectCover(mapped, aspect, widget);

      expect(back.left, closeTo(box.left, 1e-9));
      expect(back.top, closeTo(box.top, 1e-9));
      expect(back.right, closeTo(box.right, 1e-9));
      expect(back.bottom, closeTo(box.bottom, 1e-9));

      // A genuinely wide box must STAY wide after mapping (the PyroGuard
      // failure mode transposed wide boxes into tall slivers), and the mapped
      // aspect must equal the content-pixel aspect (mapping preserves shape).
      expect(mapped.width > mapped.height, isTrue,
          reason: 'a 576x128-content-px box must render wide, not tall');
      final contentPxAspect = (box.width * 720) / (box.height * 1280);
      expect(mapped.width / mapped.height, closeTo(contentPxAspect, 1e-9));
    });

    test('sensor 0 (already-upright preview) needs no aspect inversion', () {
      final aspect = uprightContentAspect(
        previewWidth: 720,
        previewHeight: 1280,
        sensorOrientation: 0,
      );
      expect(aspect, closeTo(720 / 1280, 1e-9));
    });
  });
}

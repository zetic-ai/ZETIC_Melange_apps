import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:livegate/services/frame.dart';

/// A 2x1 BGRA buffer: raw(0,0)=A(10,20,30), raw(1,0)=B(40,50,60).
FrameData twoPixelBgra(int rotationDegrees) {
  final bytes = Uint8List(2 * 1 * 4);
  bytes[0] = 10;
  bytes[1] = 20;
  bytes[2] = 30;
  bytes[3] = 255;
  bytes[4] = 40;
  bytes[5] = 50;
  bytes[6] = 60;
  bytes[7] = 255;
  return FrameData.bgra8888(
    width: 2,
    height: 1,
    bgra: bytes,
    bgraRowStride: 8,
    rotationDegrees: rotationDegrees,
  );
}

void main() {
  group('T9 orientation transform', () {
    test('rotation 0 is identity in size and pixel placement', () {
      final img = frameToUprightBgr(twoPixelBgra(0));
      expect(img.width, 2);
      expect(img.height, 1);
      expect(img.blueAt(0, 0), 10);
      expect(img.blueAt(1, 0), 40);
    });

    test('rotation 90 swaps dimensions and maps a known pixel correctly', () {
      final img = frameToUprightBgr(twoPixelBgra(90));
      // 2x1 rotated 90deg clockwise -> 1x2.
      expect(img.width, 1);
      expect(img.height, 2);
      // Upright top pixel comes from raw(0,0)=A; bottom from raw(1,0)=B.
      expect([img.blueAt(0, 0), img.greenAt(0, 0), img.redAt(0, 0)],
          [10, 20, 30]);
      expect([img.blueAt(0, 1), img.greenAt(0, 1), img.redAt(0, 1)],
          [40, 50, 60]);
    });

    test('rotation 180 keeps dimensions and reverses pixel order', () {
      final img = frameToUprightBgr(twoPixelBgra(180));
      expect(img.width, 2);
      expect(img.height, 1);
      // 180deg swaps the two horizontally: A and B trade places.
      expect(img.blueAt(0, 0), 40);
      expect(img.blueAt(1, 0), 10);
    });
  });
}

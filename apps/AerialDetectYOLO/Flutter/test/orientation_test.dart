import 'dart:ui';

import 'package:aerialdetect/widgets/coordinate_mapping.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('orientation / overlay transform', () {
    test('test_orientation_transform_roundtrip (no rotation)', () {
      // Buffer believed upright (720x1280); canvas portrait. The chosen
      // transform is BoxFit.cover with NO rotation.
      const Size image = Size(720, 1280);
      const Size canvas = Size(1080, 1920);
      const Rect src = Rect.fromLTRB(100, 300, 600, 400); // a WIDE box

      final Rect mapped = mapCoverRect(src, image, canvas);
      final Rect back = unmapCoverRect(mapped, image, canvas);

      // Round-trips to the original within tolerance.
      expect(back.left, closeTo(src.left, 1e-6));
      expect(back.top, closeTo(src.top, 1e-6));
      expect(back.right, closeTo(src.right, 1e-6));
      expect(back.bottom, closeTo(src.bottom, 1e-6));
    });

    test('a wide source box stays wide on screen (no spurious transpose)', () {
      const Size image = Size(720, 1280);
      const Size canvas = Size(1080, 1920);
      const Rect wide = Rect.fromLTRB(100, 300, 600, 400); // w=500 > h=100

      final Rect mapped = mapCoverRect(wide, image, canvas);
      expect(mapped.width, greaterThan(mapped.height),
          reason: 'a 90° transpose would turn a wide box into a tall sliver');
    });

    test('cover scale matches the larger axis ratio', () {
      const Size image = Size(720, 1280);
      const Size canvas = Size(1080, 1920);
      // max(1080/720=1.5, 1920/1280=1.5) = 1.5 (aspect matches here).
      expect(coverScale(image, canvas), closeTo(1.5, 1e-9));
    });
  });
}

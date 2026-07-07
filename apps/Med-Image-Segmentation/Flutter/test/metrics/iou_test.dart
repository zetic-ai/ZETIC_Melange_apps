import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:med_image_segmentation/metrics/iou.dart';

void main() {
  group('maskIoU', () {
    test('reports perfect agreement when both masks are empty', () {
      final pred = Uint8List.fromList([0, 0, 0, 0]);
      final golden = Uint8List.fromList([0, 0, 0, 0]);

      expect(maskIoU(pred, golden), 1.0);
    });

    test('uses intersection over union for LV cavity overlap', () {
      final pred = Uint8List.fromList([1, 1, 0, 0, 1]);
      final golden = Uint8List.fromList([1, 0, 1, 0, 1]);

      expect(maskIoU(pred, golden), 0.5);
    });
  });
}

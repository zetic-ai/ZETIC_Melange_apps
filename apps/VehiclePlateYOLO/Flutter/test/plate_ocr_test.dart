import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/plate_ocr.dart';

/// Pure-logic tests for the on-device plate OCR support code. Vision OCR
/// accuracy itself is device-only (Tier C) and NOT exercised here.
void main() {
  group('computeCropRect (bbox -> crop rect mapping + clamping)', () {
    test('pads by fraction and stays inside the image', () {
      // 100-wide x 50-tall box, 5% pad => +5 / +2.5 each side, floor/ceil out.
      final r = computeCropRect(100, 100, 200, 150, 640, 480);
      expect(r.x, 95); // floor(100 - 5)
      expect(r.y, 97); // floor(100 - 2.5)
      expect(r.width, 110); // ceil(200 + 5) = 205, 205 - 95
      expect(r.height, 56); // ceil(150 + 2.5) = 153, 153 - 97
      // Always within bounds.
      expect(r.x + r.width, lessThanOrEqualTo(640));
      expect(r.y + r.height, lessThanOrEqualTo(480));
    });

    test('clamps a box that runs past the image edges', () {
      final r = computeCropRect(-20, -10, 700, 500, 640, 480);
      expect(r.x, 0);
      expect(r.y, 0);
      expect(r.width, 640);
      expect(r.height, 480);
      expect(r.isEmpty, isFalse);
    });

    test('returns empty for degenerate or zero-size inputs', () {
      expect(computeCropRect(10, 10, 10, 10, 640, 480).isEmpty, isTrue);
      expect(computeCropRect(50, 50, 20, 20, 640, 480).isEmpty, isTrue);
      expect(computeCropRect(10, 10, 20, 20, 0, 480).isEmpty, isTrue);
    });

    test('fully out-of-bounds box collapses to empty', () {
      expect(computeCropRect(700, 10, 800, 60, 640, 480).isEmpty, isTrue);
    });
  });

  group('normalizePlateText (plate string normalization)', () {
    test('strips whitespace and punctuation from a single line', () {
      expect(normalizePlateText(['7 ABC-123\n']), '7ABC123');
    });

    test('filters punctuation/state lines and picks the plausible token', () {
      // "CA" too short to win; "•" empty; junk concat "CA7ABC123" (9) exceeds
      // the plausible max (8) so it is rejected, leaving the real plate.
      expect(normalizePlateText(['CA', '7ABC123', '•']), '7ABC123');
    });

    test('joins a genuine multi-line plate', () {
      expect(normalizePlateText(['ABC', '123']), 'ABC123');
    });

    test('returns null when nothing is plausible', () {
      expect(normalizePlateText([]), isNull);
      expect(normalizePlateText(['•', '--', ' ']), isNull);
      expect(normalizePlateText(['X']), isNull); // below min length
    });

    test('uppercases lowercase OCR output', () {
      expect(normalizePlateText(['ab12cd']), 'AB12CD');
    });
  });
}

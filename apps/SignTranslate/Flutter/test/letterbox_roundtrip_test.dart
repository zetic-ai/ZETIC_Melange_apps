
import 'package:flutter_test/flutter_test.dart';
import 'package:signtranslate/config.dart';
import 'package:signtranslate/models/text_region.dart';
import 'package:signtranslate/services/detector_preprocessor.dart';

void main() {
  const tol = 1e-3;

  void roundTrip(int srcW, int srcH, Offset p) {
    final geo = computeLetterboxGeometry(srcW, srcH);
    final model = geo.toModel(p);
    final back = geo.toFrame(model);
    expect(back.dx, closeTo(p.dx, tol),
        reason: 'x round-trip failed for $srcW x $srcH');
    expect(back.dy, closeTo(p.dy, tol),
        reason: 'y round-trip failed for $srcW x $srcH');
  }

  group('letterbox(736) forward -> inverse round-trip', () {
    test('wide frame (1280x720), points across the frame', () {
      for (final p in [
        const Offset(0, 0),
        const Offset(1279, 719),
        const Offset(640, 360),
        const Offset(17.5, 703.25),
      ]) {
        roundTrip(1280, 720, p);
      }
    });

    test('tall frame (720x1280)', () {
      roundTrip(720, 1280, const Offset(0, 0));
      roundTrip(720, 1280, const Offset(719, 1279));
      roundTrip(720, 1280, const Offset(123.4, 987.6));
    });

    test('square frame (1000x1000) and native 736', () {
      roundTrip(1000, 1000, const Offset(500, 500));
      roundTrip(736, 736, const Offset(1, 735));
    });

    test('a full quad round-trips through model space', () {
      final geo = computeLetterboxGeometry(1920, 1080);
      final quad = Quad(
        const Offset(100, 200),
        const Offset(400, 210),
        const Offset(390, 300),
        const Offset(95, 290),
      );
      final back = quad.map(geo.toModel).map(geo.toFrame);
      for (var i = 0; i < 4; i++) {
        expect((back.points[i] - quad.points[i]).distance, lessThan(tol));
      }
    });
  });

  group('inverse order is exactly reversed (subtract pad, THEN divide)', () {
    test('a wrong-order inverse fails on an off-center point', () {
      final geo = computeLetterboxGeometry(1280, 720); // padY=161, scale=0.575
      const framePt = Offset(200, 100);
      final modelPt = geo.toModel(framePt);

      // Correct inverse.
      final correct = geo.toFrame(modelPt);
      expect(correct.dx, closeTo(200, tol));
      expect(correct.dy, closeTo(100, tol));

      // WRONG order (divide first, then subtract pad) — must differ wildly,
      // proving the round-trip test discriminates inverse ordering.
      final wrongY = modelPt.dy / geo.scale - geo.padY;
      expect((wrongY - framePt.dy).abs(), greaterThan(1.0));
    });

    test('pad-boundary points map to frame edges', () {
      final geo = computeLetterboxGeometry(1280, 720);
      // Model y = padY is the first content row -> frame y = 0.
      final top = geo.toFrame(Offset(0, geo.padY.toDouble()));
      expect(top.dy, closeTo(0, tol));
      // Model y = padY + 720*scale -> frame y = 720.
      final bottom =
          geo.toFrame(Offset(0, geo.padY + 720 * geo.scale));
      expect(bottom.dy, closeTo(720, tol));
    });
  });

  test('736 constant is pinned (not 640)', () {
    expect(kDetInputSize, 736);
    final geo = computeLetterboxGeometry(1472, 1472);
    expect(geo.scale, closeTo(0.5, 1e-9));
  });
}

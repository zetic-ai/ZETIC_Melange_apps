import 'package:flutter_test/flutter_test.dart';
import 'package:vehicleplateyolo/services/orientation.dart';

/// TRAP: the PyroGuard bug was a SPURIOUS rotation transposing every box. The
/// reference iOS BGRA buffer arrives upright (720x1280) -> rotation 0 -> the
/// transform must be IDENTITY. Android YUV420 is sensor-landscape -> 90 (Tier C
/// device calibration). Either way the chosen transform must round-trip.
void main() {
  test('rotation 0 (iOS upright) is identity and round-trips', () {
    const o = FrameOrientation(0);
    expect(o.isIdentity, isTrue);
    expect(o.uprightWidth(720, 1280), 720);
    expect(o.uprightHeight(720, 1280), 1280);
    expect(o.uprightToBuffer(100, 200, 720, 1280), [100, 200]);
    final back = o.bufferToUpright(100, 200, 720, 1280);
    expect(back, [100, 200]);
  });

  test('rotation 90 (Android landscape) round-trips a known pixel', () {
    const o = FrameOrientation(90);
    // Sensor-landscape buffer; upright is portrait.
    expect(o.uprightWidth(1280, 720), 720);
    expect(o.uprightHeight(1280, 720), 1280);
    for (final p in [
      [0, 0],
      [123, 456],
      [719, 1279],
    ]) {
      final buf = o.uprightToBuffer(p[0], p[1], 1280, 720);
      final up = o.bufferToUpright(buf[0], buf[1], 1280, 720);
      expect(up, p, reason: 'transform must round-trip');
    }
  });
}

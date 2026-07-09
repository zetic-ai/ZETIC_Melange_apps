import 'dart:io';
import 'dart:typed_data';

import 'package:dentalxraydetect/services/preprocessor.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

/// Integration test on a REAL demo radiograph: decode -> bilinear letterbox ->
/// NCHW, asserting the preprocessing produces a well-formed model input. This
/// exercises the actual `img.copyResize(Interpolation.linear)` path on a genuine
/// panoramic X-ray (val_28.png, 2872x1504) — the model.run step itself is
/// device-only and not covered here.
///
/// The app itself is upload-only (no bundled samples), so this fixture is loaded
/// from the repo `demo_images/` directory (a sibling of the Flutter project),
/// NOT from the app asset bundle.
void main() {
  test('val_28 demo radiograph preprocesses to a valid 640 NCHW tensor', () {
    final File f = File('../demo_images/val_28.png');
    expect(f.existsSync(), isTrue,
        reason: 'repo demo image ../demo_images/val_28.png must exist');

    final img.Image decoded = img.decodeImage(f.readAsBytesSync())!;
    final img.Image oriented = img.bakeOrientation(decoded);
    final Uint8List rgb = oriented.getBytes(order: img.ChannelOrder.rgb);
    final int w = oriented.width, h = oriented.height;

    // val_28 is a wide panoramic: width-limited letterbox -> scaledW == 640.
    final LetterboxParams p = computeLetterbox(w, h);
    expect(p.target, 640);
    expect(p.scaledW, 640);
    expect(p.padY, greaterThan(0)); // vertical gray bars top & bottom

    final Float32List out = Float32List(kInputElements);
    letterboxRgbToNchw(rgb, w, h, p, out);

    // 1) Exact NCHW element count.
    expect(out.length, 3 * kInputSize * kInputSize);

    // 2) Every value normalized to [0, 1].
    double lo = 1e9, hi = -1e9;
    for (final double v in out) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    expect(lo, greaterThanOrEqualTo(0.0));
    expect(hi, lessThanOrEqualTo(1.0));

    // 3) The top row (dy=0) sits in the pad band -> gray kPadValue on all planes.
    const int channelStride = kInputSize * kInputSize;
    expect(out[0], closeTo(kPadValue, 1e-6));
    expect(out[channelStride], closeTo(kPadValue, 1e-6));
    expect(out[2 * channelStride], closeTo(kPadValue, 1e-6));

    // 4) A central band carries real image content, not pad: some pixels differ
    //    from kPadValue (bilinear-resized radiograph, not a flat fill).
    final int midRow = kInputSize ~/ 2;
    int nonPad = 0;
    for (int dx = 0; dx < kInputSize; dx++) {
      if ((out[midRow * kInputSize + dx] - kPadValue).abs() > 1e-3) nonPad++;
    }
    expect(nonPad, greaterThan(kInputSize ~/ 4),
        reason: 'center row should be mostly real image content');
  });
}

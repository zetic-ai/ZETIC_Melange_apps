import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// Live preprocessing: display frames are already center-cropped square, so this
/// resizes to [size] with bilinear interpolation, scales to [0, 1], and packs NCHW RGB,
/// written into the reused [out] buffer (length 3*size*size). Used only to time a
/// representative preprocess cost; the model consumes the bundled golden input so
/// the IoU stays about quantization drift.
void packNCHW(img.Image src, int size, Float32List out) {
  final r = (src.width == size && src.height == size)
      ? src
      : img.copyResize(
          src,
          width: size,
          height: size,
          interpolation: img.Interpolation.linear,
        );
  final plane = size * size;
  var idx = 0;
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final p = r.getPixel(x, y);
      out[idx] = p.rNormalized.toDouble();
      out[plane + idx] = p.gNormalized.toDouble();
      out[2 * plane + idx] = p.bNormalized.toDouble();
      idx++;
    }
  }
}

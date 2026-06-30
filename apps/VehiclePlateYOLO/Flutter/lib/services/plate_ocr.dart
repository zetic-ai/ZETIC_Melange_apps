import 'dart:io' show Platform;
import 'dart:math' as math;

import 'package:flutter/services.dart';

/// On-device license-plate OCR.
///
/// The Melange model is detection-only; this is a separate, on-device read step
/// that crops a detected plate from the live BGRA frame and runs **Apple Vision**
/// text recognition through the `platehawk/ocr` MethodChannel (see the native
/// `PlateOcrPlugin` in `ios/Runner/AppDelegate.swift`).
///
/// iOS-first: on Android (no Vision) [recognize] is a graceful no-op returning
/// `null`, so the app still builds and runs (boxes only, no plate text).
///
/// The crop-rect mapping ([computeCropRect]) and the text normalization
/// ([normalizePlateText]) are pure functions so they can be unit-tested on the
/// host without a device. Vision accuracy itself is device-only (Tier C).
class PlateOcr {
  PlateOcr._();

  static const MethodChannel _channel = MethodChannel('platehawk/ocr');

  /// Minimum crop side (px) below which the crop is nearest-neighbour upscaled
  /// to give Vision more pixels to work with on small/distant plates.
  static const int _minCropSide = 48;

  /// Crop [left,top,right,bottom] (upright source-image pixels) from a BGRA
  /// frame and run on-device OCR. Returns a normalized plate string, or `null`
  /// if nothing confident / off-iOS.
  ///
  /// NOTE: the crop assumes upright-image space == buffer space, which holds for
  /// the iOS BGRA hot path (rotation 0). Callers must only invoke this when that
  /// holds (the camera screen guards on `rotation == 0` and `Platform.isIOS`).
  static Future<String?> recognize({
    required Uint8List bgra,
    required int bytesPerRow,
    required int width,
    required int height,
    required double left,
    required double top,
    required double right,
    required double bottom,
  }) async {
    if (!Platform.isIOS) return null; // Android: no Vision — graceful skip.

    final rect = computeCropRect(left, top, right, bottom, width, height);
    if (rect.isEmpty) return null;

    final crop = _cropBgraToRgba(bgra, bytesPerRow, rect);
    final scaled = _maybeUpscale(crop, rect.width, rect.height);

    try {
      final res = await _channel.invokeMethod<List<dynamic>>('recognize', {
        'rgba': scaled.bytes,
        'width': scaled.width,
        'height': scaled.height,
      });
      if (res == null) return null;
      final lines = <String>[];
      for (final item in res) {
        final m = item as Map;
        final text = m['text'] as String?;
        if (text != null) lines.add(text);
      }
      return normalizePlateText(lines);
    } on PlatformException {
      return null;
    } on MissingPluginException {
      return null; // native handler not registered (e.g. unexpected platform).
    }
  }

  /// Crop the [rect] region out of a BGRA buffer into a tightly-packed RGBA
  /// buffer (alpha forced opaque). [bytesPerRow] is the BGRA row stride (may
  /// exceed `width * 4` due to padding).
  static Uint8List _cropBgraToRgba(
    Uint8List src,
    int bytesPerRow,
    CropRect rect,
  ) {
    final out = Uint8List(rect.width * rect.height * 4);
    var o = 0;
    for (var y = 0; y < rect.height; y++) {
      var s = (rect.y + y) * bytesPerRow + rect.x * 4;
      for (var x = 0; x < rect.width; x++) {
        out[o] = src[s + 2]; // R (BGRA -> RGBA)
        out[o + 1] = src[s + 1]; // G
        out[o + 2] = src[s]; // B
        out[o + 3] = 255; // A (opaque)
        s += 4;
        o += 4;
      }
    }
    return out;
  }

  /// Nearest-neighbour upscale of small crops to [_minCropSide] on the short
  /// side. Passes the crop through unchanged when already large enough.
  static _Rgba _maybeUpscale(Uint8List bytes, int w, int h) {
    final shortSide = math.min(w, h);
    if (shortSide >= _minCropSide) return _Rgba(bytes, w, h);
    final f = (_minCropSide / shortSide).ceil();
    final nw = w * f;
    final nh = h * f;
    final out = Uint8List(nw * nh * 4);
    for (var y = 0; y < nh; y++) {
      final sy = y ~/ f;
      for (var x = 0; x < nw; x++) {
        final sx = x ~/ f;
        final si = (sy * w + sx) * 4;
        final di = (y * nw + x) * 4;
        out[di] = bytes[si];
        out[di + 1] = bytes[si + 1];
        out[di + 2] = bytes[si + 2];
        out[di + 3] = bytes[si + 3];
      }
    }
    return _Rgba(out, nw, nh);
  }
}

/// Integer crop rectangle in image-pixel space.
class CropRect {
  const CropRect(this.x, this.y, this.width, this.height);

  final int x;
  final int y;
  final int width;
  final int height;

  bool get isEmpty => width <= 0 || height <= 0;

  @override
  String toString() => 'CropRect($x,$y,${width}x$height)';
}

/// Pure: map a detection bbox (upright source-image pixels) to an integer crop
/// rect, padded by [padFraction] of the box size on each side and clamped to the
/// image bounds. Returns an empty rect for degenerate/out-of-bounds input.
CropRect computeCropRect(
  double left,
  double top,
  double right,
  double bottom,
  int imageWidth,
  int imageHeight, {
  double padFraction = 0.05,
}) {
  if (imageWidth <= 0 || imageHeight <= 0) return const CropRect(0, 0, 0, 0);

  final bw = right - left;
  final bh = bottom - top;
  if (bw <= 0 || bh <= 0) return const CropRect(0, 0, 0, 0);

  final px = bw * padFraction;
  final py = bh * padFraction;

  var l = (left - px).floor();
  var t = (top - py).floor();
  var r = (right + px).ceil();
  var b = (bottom + py).ceil();

  if (l < 0) l = 0;
  if (t < 0) t = 0;
  if (r > imageWidth) r = imageWidth;
  if (b > imageHeight) b = imageHeight;

  if (r <= l || b <= t) return const CropRect(0, 0, 0, 0);
  return CropRect(l, t, r - l, b - t);
}

/// Pure: normalize raw Vision text lines into a single plate string.
///
/// Rule: uppercase each line, strip everything except `A-Z0-9` (drops spaces &
/// punctuation), then consider each stripped line *and* the in-order
/// concatenation of all lines as candidates. Keep the **longest plausible**
/// candidate, where plausible = length in `[minLength, maxLength]`. Returns
/// `null` if nothing plausible.
///
/// Examples: `["7 ABC-123\n"]` -> `"7ABC123"`; `["CA", "7ABC123", "•"]` ->
/// `"7ABC123"` (state abbreviation / punctuation filtered out, junk
/// concatenation rejected by [maxLength]); `["ABC", "123"]` -> `"ABC123"`
/// (multi-line plate joined).
String? normalizePlateText(
  List<String> lines, {
  int minLength = 2,
  int maxLength = 8,
}) {
  final candidates = <String>[];
  final joined = StringBuffer();
  for (final line in lines) {
    final t = _stripAlnum(line);
    if (t.isEmpty) continue;
    candidates.add(t);
    joined.write(t);
  }
  final all = joined.toString();
  if (all.isNotEmpty) candidates.add(all);

  String? best;
  for (final c in candidates) {
    if (c.length < minLength || c.length > maxLength) continue;
    if (best == null || c.length > best.length) best = c;
  }
  return best;
}

final RegExp _nonAlnum = RegExp(r'[^A-Z0-9]');

String _stripAlnum(String s) => s.toUpperCase().replaceAll(_nonAlnum, '');

/// Packed RGBA image (width*height*4 bytes).
class _Rgba {
  _Rgba(this.bytes, this.width, this.height);

  final Uint8List bytes;
  final int width;
  final int height;
}

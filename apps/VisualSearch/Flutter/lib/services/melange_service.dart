import 'dart:ui' show Rect;

import 'package:flutter/foundation.dart';
import 'package:zetic_mlange/zetic_mlange.dart';

import '../models/detection.dart';
import '../models/gallery_item.dart';
import 'gallery.dart';
import 'model_registry.dart';
import 'postprocessor.dart';
import 'preprocessor.dart';

/// Personal key placeholder. Filled at build time by `adapt_mlange_key.sh`
/// (matches its `const _sourceMlangeKey = "YOUR_MLANGE_KEY"` substitution) —
/// a REAL key must NEVER be committed. The key is embedded in the client.
// ignore: prefer_single_quotes
const _sourceMlangeKey = "YOUR_MLANGE_KEY";

/// Result of one full snap→search pass, with the diagnostics the release HUD
/// needs (Dart print does not reach the device console in release builds).
class SearchOutcome {
  const SearchOutcome({
    required this.primary,
    required this.cropRect,
    required this.frameWidth,
    required this.frameHeight,
    required this.results,
    required this.detectMs,
    required this.embedMs,
    required this.totalMs,
  });

  /// Primary salient box (normalized 0..1), or null if the detector found
  /// nothing (→ center-crop fallback was embedded).
  final Detection? primary;

  /// Crop actually embedded, normalized 0..1 in the frame.
  final Rect cropRect;
  final int frameWidth;
  final int frameHeight;

  final List<SearchResult> results;
  final int detectMs;
  final int embedMs;
  final int totalMs;

  bool get usedFallback => primary == null;
}

/// Owns the TWO-model Melange lifecycle (detect + embed) and the full
/// pure-Dart pipeline that stitches them: decode → detect → crop → embed →
/// rank. See HANDOFF.md "Dual-model lifecycle notes".
class MelangeService {
  MelangeService();

  ZeticMLangeModel? _detect;
  ZeticMLangeModel? _embed;
  Gallery? _gallery;

  bool get isReady =>
      _detect != null &&
      !_detect!.isClosed &&
      _embed != null &&
      !_embed!.isClosed &&
      _gallery != null;

  Gallery? get gallery => _gallery;

  /// Download (if needed), initialize, and warm up BOTH models, and load the
  /// bundled gallery. [onProgress] drives the loading bar (0..1 across both
  /// downloads). Load order: detect first (smaller, 10.7 MB), then embed
  /// (45.8 MB) — see the lifecycle notes in HANDOFF.md.
  Future<void> init({void Function(double progress, String stage)? onProgress}) async {
    if (isReady) return;

    onProgress?.call(0, 'Loading detector…');
    _detect ??= await ZeticMLangeModel.create(
      personalKey: _sourceMlangeKey,
      name: ModelRegistry.detectName,
      version: ModelRegistry.detectVersion,
      modelMode: ModelMode.runAuto,
      onProgress: (p) => onProgress?.call(p * 0.45, 'Loading detector…'),
    );

    onProgress?.call(0.45, 'Loading embedder…');
    _embed ??= await ZeticMLangeModel.create(
      personalKey: _sourceMlangeKey,
      name: ModelRegistry.embedName,
      version: ModelRegistry.embedVersion,
      modelMode: ModelMode.runAuto,
      onProgress: (p) => onProgress?.call(0.45 + p * 0.45, 'Loading embedder…'),
    );

    onProgress?.call(0.92, 'Loading catalog…');
    _gallery ??= await Gallery.load();

    onProgress?.call(0.96, 'Warming up…');
    _warmUp();
    onProgress?.call(1.0, 'Ready');
  }

  /// One dummy inference per model so the first real snap is not the slow one.
  void _warmUp() {
    final d = _detect, e = _embed;
    if (d != null && !d.isClosed) {
      final zeros = Float32List(3 * kDetectSize * kDetectSize);
      final o = d.run([Tensor.float32List(zeros, shape: const [1, 3, kDetectSize, kDetectSize])]);
      o.first.asFloat32List();
    }
    if (e != null && !e.isClosed) {
      final zeros = Float32List(3 * kEmbedSize * kEmbedSize);
      final o = e.run([Tensor.float32List(zeros, shape: const [1, 3, kEmbedSize, kEmbedSize])]);
      o.first.asFloat32List();
    }
  }

  /// Run the full snap→search pipeline on a captured JPEG. CPU-heavy Dart
  /// (decode, letterbox, decode-post, crop/resize) runs off-thread via
  /// [compute]; both native `model.run` calls run on the main isolate because
  /// the SDK binds each model handle to the isolate that created it.
  Future<SearchOutcome> search(Uint8List jpegBytes, {int topK = 5}) async {
    final detect = _detect, embed = _embed, gallery = _gallery;
    if (detect == null || detect.isClosed || embed == null || embed.isClosed || gallery == null) {
      throw StateError('MelangeService.search called before init()');
    }

    final total = Stopwatch()..start();

    // --- DETECT: decode + letterbox (isolate) → run (main) → decode (isolate)
    final detectSw = Stopwatch()..start();
    final DetectBundle pre = await compute(decodeAndPreprocessDetect, jpegBytes);
    final detOut = detect.run([
      Tensor.float32List(pre.input, shape: const [1, 3, kDetectSize, kDetectSize]),
    ]);
    // asFloat32List() is a view over a reused native buffer — copy before the
    // next run overwrites it and before crossing the isolate boundary.
    final rawDet = Float32List.fromList(detOut.first.asFloat32List());
    final dets = await compute(
      postprocessDetect,
      PostprocessRequest(
        output: rawDet,
        scale: pre.scale,
        padX: pre.padX,
        padY: pre.padY,
        srcWidth: pre.frame.width,
        srcHeight: pre.frame.height,
      ),
    );
    final Detection? primary = primaryBox(dets);
    detectSw.stop();

    // --- EMBED: crop + resize (isolate) → run (main) → rank (main)
    final embedSw = Stopwatch()..start();
    final EmbedBundle emb = await compute(
      preprocessEmbed,
      EmbedRequest(
        width: pre.frame.width,
        height: pre.frame.height,
        rgb: pre.frame.rgb,
        box: primary?.rect, // null → center-crop fallback (embed never skipped)
      ),
    );
    final embOut = embed.run([
      Tensor.float32List(emb.input, shape: const [1, 3, kEmbedSize, kEmbedSize]),
    ]);
    final Float32List vec = Float32List.fromList(embOut.first.asFloat32List());
    final results = gallery.topK(vec, k: topK);
    embedSw.stop();
    total.stop();

    // Normalize the crop rect to 0..1 for size-independent UI drawing.
    final Rect cropNorm = Rect.fromLTRB(
      emb.cropRect.left / pre.frame.width,
      emb.cropRect.top / pre.frame.height,
      emb.cropRect.right / pre.frame.width,
      emb.cropRect.bottom / pre.frame.height,
    );

    return SearchOutcome(
      primary: primary,
      cropRect: cropNorm,
      frameWidth: pre.frame.width,
      frameHeight: pre.frame.height,
      results: results,
      detectMs: detectSw.elapsedMilliseconds,
      embedMs: embedSw.elapsedMilliseconds,
      totalMs: total.elapsedMilliseconds,
    );
  }

  void dispose() {
    if (_detect != null && !_detect!.isClosed) _detect!.close();
    if (_embed != null && !_embed!.isClosed) _embed!.close();
    _detect = null;
    _embed = null;
  }
}

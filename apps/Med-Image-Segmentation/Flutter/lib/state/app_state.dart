import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:image/image.dart' as img;

import '../assets_loader.dart';
import '../benchmark/benchmark_stats.dart';
import '../benchmark/device_info.dart';
import '../benchmark/memory_probe.dart';
import '../inference/inference_client.dart';
import '../metrics/iou.dart';
import '../overlay/mask_overlay.dart';
import '../pipeline/preprocess.dart';
import '../video/clip_source.dart';

enum Phase { loadingAssets, downloadingModel, running, error }

const _modelName = 'realtonypark/EchoNet-DeepLab-v3';
const _sourceMlangeKey = "YOUR_MLANGE_KEY";
const _dartDefineMlangeKey = String.fromEnvironment('MLANGE_KEY');

String get _mlangeKey =>
    _dartDefineMlangeKey.isNotEmpty ? _dartDefineMlangeKey : _sourceMlangeKey;

bool get _isMlangeKeyConfigured =>
    _mlangeKey.isNotEmpty && _mlangeKey != 'YOUR_MLANGE_KEY';

/// Decoded display + input-size frames for one clip.
class _ClipImages {
  _ClipImages(this.display, this.inputSize);
  final List<img.Image> display;
  final List<img.Image> inputSize;
}

/// Orchestrates asset load → model load → the per-frame loop, and exposes the
/// state the UI renders. Inference runs in a worker isolate; everything here is
/// on the UI isolate (render + benchmark aggregation).
class AppState extends ChangeNotifier {
  Phase phase = Phase.loadingAssets;
  double downloadProgress = 0;
  String errorMessage = '';

  ClipLibrary? _library;
  final List<_ClipImages> _images = [];
  DeviceIdentity? device;

  int activeClip = 0;
  List<ClipData> get clips => _library?.clips ?? const [];
  ClipData? get currentClip =>
      _library == null ? null : _library!.clips[activeClip];
  String get attribution => currentClip?.attribution ?? '';
  String? get currentFramePath =>
      currentClip == null ? null : currentClip!.framePaths[frameIndex];

  final _client = InferenceClient();
  var _stats = BenchmarkStats();
  ClipSource? _source;

  /// Fires once per rendered frame. The video + HUD listen to THIS (not the main
  /// ChangeNotifier) so the interactive controls (clip selector, pacing toggle)
  /// are NOT rebuilt ~28x/sec — which otherwise drops taps on slower devices.
  final ValueNotifier<int> frameTick = ValueNotifier<int>(0);

  // Per-frame UI state.
  int frameIndex = 0;
  ui.Image? maskImage;
  int cavityAreaPx = 0;
  BenchmarkSnapshot snapshot = const BenchmarkSnapshot();

  // Fractional-area-change (EF proxy) over a rolling ~2s of cavity areas.
  final Queue<int> _areaHistory = Queue<int>();
  double facPercent = 0;

  PacingMode get pacing => _source?.mode ?? PacingMode.benchmark;
  int _frame = 0; // frame index within the active clip
  int _memTick = 0;
  double _lastMemoryMB = 0;

  // Reused per-frame scratch buffers (input dims are shared across clips).
  late final Uint8List _maskBuf;
  late final Uint8List _rgbaBuf;
  late final Float32List _preprocBuf;

  Future<void> init() async {
    try {
      final lib = await ClipLibrary.load();
      _library = lib;
      _maskBuf = Uint8List(lib.inputH * lib.inputW);
      _rgbaBuf = Uint8List(lib.inputH * lib.inputW * 4);
      _preprocBuf = Float32List(lib.inputC * lib.inputH * lib.inputW);

      for (final clip in lib.clips) {
        final display = <img.Image>[];
        final inputSize = <img.Image>[];
        for (final p in clip.framePaths) {
          final bytes = await rootBundle.load(p);
          final decoded = img.decodePng(bytes.buffer.asUint8List())!;
          display.add(decoded);
          inputSize.add(
            img.copyResize(
              decoded,
              width: clip.inputW,
              height: clip.inputH,
              interpolation: img.Interpolation.linear,
            ),
          );
        }
        _images.add(_ClipImages(display, inputSize));
      }
      device = await DeviceIdentity.load();

      if (!_isMlangeKeyConfigured) {
        _fail(
          'Melange key missing. Run ./adapt_mlange_key.sh from the repo root, or pass --dart-define=MLANGE_KEY=YOUR_KEY.',
        );
        return;
      }

      phase = Phase.downloadingModel;
      notifyListeners();

      _client.onProgress = (p) {
        downloadProgress = p;
        notifyListeners();
      };
      await _client.start(
        personalKey: _mlangeKey,
        modelName: _modelName,
        inputsByClip: lib.inputsByClip,
        c: lib.inputC,
        h: lib.inputH,
        w: lib.inputW,
      );

      phase = Phase.running;
      notifyListeners();
      debugPrint(
        '[medseg] model ready: $_modelName (${lib.clips.length} clips)',
      );

      _source = ClipSource(fps: 24)..onTick = _tick;
      unawaited(_source!.start());
    } catch (e) {
      _fail('$e');
    }
  }

  Future<void> _tick() async {
    final lib = _library!;
    final clipIdx = activeClip; // capture: may change during the await
    final clip = lib.clips[clipIdx];
    final i = _frame % clip.numFrames;
    _frame++;

    // 1) Inference on the golden input (worker isolate, timed there).
    final result = await _client.run(clipIdx, i);
    // If the user switched clips mid-inference, drop this stale frame.
    if (clipIdx != activeClip) return;
    final logits = result.logits;

    // 2) Threshold -> mask (reused buffer); IoU vs golden; cavity area.
    final maskLen = clip.maskLen;
    final mask = _maskBuf;
    var area = 0;
    for (var k = 0; k < maskLen; k++) {
      if (logits[k] > 0) {
        mask[k] = 1;
        area++;
      } else {
        mask[k] = 0;
      }
    }
    final iou = maskIoU(mask, clip.goldenMasks[i]);

    // 3) Live preprocess timing (normalize + NCHW pack into a reused buffer;
    // result unused — the model runs on the bundled golden input).
    final psw = Stopwatch()..start();
    packNCHW(_images[clipIdx].inputSize[i], clip.inputW, _preprocBuf);
    psw.stop();

    // 4) Memory (throttled — a channel roundtrip per frame would skew FPS).
    if (_memTick++ % 15 == 0) _lastMemoryMB = await MemoryProbe.footprintMB();

    _stats.record(
      inferenceMs: result.inferenceMs,
      preprocessMs: psw.elapsedMicroseconds / 1000.0,
      iou: iou,
      memoryMB: _lastMemoryMB,
    );

    // 5) EF-proxy: fractional area change over a rolling window.
    _areaHistory.addLast(area);
    while (_areaHistory.length > 48) {
      _areaHistory.removeFirst();
    }
    final mx = _areaHistory.reduce((a, b) => a > b ? a : b);
    final mn = _areaHistory.reduce((a, b) => a < b ? a : b);
    facPercent = mx > 0 ? (mx - mn) / mx * 100 : 0;

    // 6) Overlay + publish. Dispose the previous mask image only AFTER the next
    // frame renders (when it is off-screen) — disposing an on-screen image
    // silently fails to free native memory.
    final imgMask = await buildMaskImage(
      mask,
      clip.inputW,
      clip.inputH,
      _rgbaBuf,
    );
    if (clipIdx != activeClip) {
      imgMask.dispose();
      return;
    }
    final old = maskImage;
    maskImage = imgMask;
    if (old != null) {
      WidgetsBinding.instance.addPostFrameCallback((_) => old.dispose());
    }
    frameIndex = i;
    cavityAreaPx = area;
    snapshot = _stats.snapshot();
    // Per-frame update: bump the frame notifier (rebuilds only video + HUD), NOT
    // notifyListeners() (which would rebuild the whole tree incl. the controls).
    frameTick.value++;
  }

  /// Switch the active demo clip. Resets the benchmark so each clip shows its
  /// own numbers; the single pacer loop keeps running.
  void selectClip(int index) {
    if (index == activeClip || index < 0 || index >= clips.length) return;
    activeClip = index;
    _frame = 0;
    frameIndex = 0;
    _stats = BenchmarkStats();
    _areaHistory.clear();
    facPercent = 0;
    cavityAreaPx = 0;
    snapshot = const BenchmarkSnapshot();
    maskImage?.dispose();
    maskImage = null;
    notifyListeners();
  }

  void setPacing(PacingMode m) {
    _source?.mode = m;
    notifyListeners();
  }

  void _fail(String msg) {
    phase = Phase.error;
    errorMessage = msg;
    notifyListeners();
  }

  @override
  void dispose() {
    _source?.stop();
    _client.dispose();
    maskImage?.dispose();
    frameTick.dispose();
    super.dispose();
  }
}

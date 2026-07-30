import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/text_region.dart';
import '../services/ctc_decoder.dart';
import '../services/frame_data.dart';
import '../services/frame_scheduler.dart';
import '../services/melange_service.dart';
import '../services/pipeline_isolate.dart';
import '../services/region_tracker.dart';
import '../theme.dart';
import '../widgets/hud_bar.dart';
import '../widgets/text_overlay.dart';

/// Live scene-text reading: camera feed + region-pinned text overlay + HUD.
///
/// Per processed frame (scheduler-gated):
///  * detection pass — frame → pipeline isolate (BGR + letterbox-736) →
///    detector on main → DB postprocess in isolate → tracker update →
///    top-K cache-miss recognition (deskew + rec-preprocess in isolate,
///    recognizer + CTC decode on main);
///  * staggered pass — leftover cache-misses recognized from the isolate's
///    RETAINED frame between detection frames;
///  * frames arriving while a pass is in flight are DROPPED, never queued.
class MainScreen extends StatefulWidget {
  const MainScreen({
    super.key,
    required this.service,
    required this.pipeline,
    required this.decoder,
  });

  final MelangeService service;
  final PipelineWorker pipeline;
  final CtcDecoder decoder;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  CameraDescription? _camera;
  bool _streaming = false;
  String? _permissionError;

  final FrameScheduler _scheduler = FrameScheduler();
  final RegionTracker _tracker = RegionTracker();

  /// Cache-miss regions awaiting staggered recognition on upcoming frames
  /// (already priority-sorted; crops come from the isolate's retained frame).
  List<TrackedRegion> _pendingRecognition = [];

  List<RecognizedRegion> _display = const [];
  int _revision = 0;

  HudStats _stats = const HudStats();
  double _fps = 0;
  int _lastFrameUs = 0;

  // Upright frame dimensions (the coordinate space of every tracked quad).
  double _frameW = 0;
  double _frameH = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _setupCamera();
  }

  Future<void> _setupCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _permissionError = 'No camera found on this device.');
        return;
      }
      _camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final controller = CameraController(
        _camera!,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: defaultTargetPlatform == TargetPlatform.iOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.yuv420,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      _controller = controller;
      setState(() => _permissionError = null);
      await _startStream();
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() {
        _permissionError = (e.code == 'CameraAccessDenied' ||
                e.code == 'CameraAccessDeniedWithoutPrompt')
            ? 'Camera permission denied. Enable it in Settings to use GlyphGo.'
            : 'Camera error: ${e.description ?? e.code}';
      });
    }
  }

  Future<void> _startStream() async {
    final controller = _controller;
    if (controller == null || _streaming) return;
    await controller.startImageStream(_onFrame);
    _streaming = true;
  }

  Future<void> _stopStream() async {
    final controller = _controller;
    if (controller == null || !_streaming) return;
    _streaming = false;
    try {
      await controller.stopImageStream();
    } catch (_) {/* ignore */}
  }

  void _onFrame(CameraImage image) {
    _tickFps();
    if (!mounted) return;
    // Busy guard (mandated behavior 4): drop, never queue.
    if (!_scheduler.tryBeginPass()) return;
    unawaited(_process(image));
  }

  void _tickFps() {
    final nowUs = DateTime.now().microsecondsSinceEpoch;
    if (_lastFrameUs != 0) {
      final dtMs = (nowUs - _lastFrameUs) / 1000.0;
      if (dtMs > 0) {
        final inst = 1000.0 / dtMs;
        _fps = _fps == 0 ? inst : _fps * 0.9 + inst * 0.1;
      }
    }
    _lastFrameUs = nowUs;
  }

  Future<void> _process(CameraImage image) async {
    // Copy the camera planes BEFORE the first await — the plugin recycles
    // the buffer as soon as this callback chain yields.
    final rotation = defaultTargetPlatform == TargetPlatform.iOS
        ? 0
        : (_camera?.sensorOrientation ?? 0);
    final frame = FrameData.fromCameraImage(image, rotationDegrees: rotation);

    var cropsRun = 0;
    var detModelMs = _stats.detectorMs;
    try {
      if (_scheduler.shouldRunDetection()) {
        final passSw = Stopwatch()..start();

        final prep = await widget.pipeline.prepareDetection(frame);
        _frameW = prep.geometry.srcWidth.toDouble();
        _frameH = prep.geometry.srcHeight.toDouble();

        final detSw = Stopwatch()..start();
        final heatmap = widget.service.runDetector(prep.input);
        detSw.stop();
        detModelMs = detSw.elapsedMilliseconds;

        final db = await widget.pipeline.postprocessDetection(heatmap);
        final update = _tracker.update(db.quads);

        // Mandated behavior 1 (top-K) + 2 (staggered cache): recognize the
        // K highest-priority misses now, queue the rest for later frames.
        final prioritized = _scheduler.selectForRecognition(
          update.misses,
          limit: update.misses.length,
        );
        final now = prioritized.length <= _scheduler.topK
            ? prioritized
            : prioritized.sublist(0, _scheduler.topK);
        _pendingRecognition = prioritized.length <= _scheduler.topK
            ? []
            : prioritized.sublist(_scheduler.topK);

        cropsRun = await _recognize(now);

        passSw.stop();
        _scheduler.recordDetectionPass(
          passMs: passSw.elapsedMilliseconds,
          modelMs: detModelMs,
        );

        _publish(
          cropsRun: cropsRun,
          detModelMs: detModelMs,
          heatMin: db.mapMin,
          heatMax: db.mapMax,
          heatMean: db.mapMean,
        );
      } else if (_pendingRecognition.isNotEmpty) {
        // Staggered recognition between detection frames, from the retained
        // frame — still capped at top-K per frame.
        final batch = _scheduler.selectForRecognition(_pendingRecognition);
        _pendingRecognition = [
          for (final r in _pendingRecognition)
            if (!batch.contains(r)) r,
        ];
        cropsRun = await _recognize(batch);
        _publish(cropsRun: cropsRun, detModelMs: detModelMs);
      }
      // else: idle frame — cached overlay keeps displaying as-is.
    } catch (e) {
      debugPrint('GlyphGo pipeline error: $e');
    } finally {
      _scheduler.endPass();
    }
  }

  Future<int> _recognize(List<TrackedRegion> regions) async {
    var crops = 0;
    for (final region in regions) {
      if (!_tracker.isAlive(region)) continue;
      final input = await widget.pipeline.prepareCrop(region.quad);
      final sw = Stopwatch()..start();
      final output = widget.service.runRecognizer(input);
      sw.stop();
      _scheduler.recordRecognition(sw.elapsedMilliseconds);
      final result = widget.decoder.decode(output);
      _tracker.commit(region, result.text, result.confidence);
      crops++;
    }
    return crops;
  }

  void _publish({
    required int cropsRun,
    required int detModelMs,
    double? heatMin,
    double? heatMax,
    double? heatMean,
  }) {
    if (!mounted) return;
    _display = _tracker.displayRegions();
    _revision++;
    setState(() {
      _stats = HudStats(
        detectorMs: detModelMs,
        detectorEmaMs: _scheduler.emaDetectorModelMs,
        recognizerMsPerCrop: _scheduler.emaRecognizerModelMs,
        cropsThisFrame: cropsRun,
        regionsRead: _display.length,
        fps: _fps,
        droppedFrames: _scheduler.droppedFrames,
        bufferWidth: _frameW.toInt(),
        bufferHeight: _frameH.toInt(),
        heatmapMin: heatMin ?? _stats.heatmapMin,
        heatmapMax: heatMax ?? _stats.heatmapMax,
        heatmapMean: heatMean ?? _stats.heatmapMean,
      );
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _stopStream();
    } else if (state == AppLifecycleState.resumed) {
      _startStream();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _stopStream();
    _controller?.dispose();
    widget.pipeline.dispose();
    widget.service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_permissionError != null) {
      return _PermissionView(message: _permissionError!, onRetry: _setupCamera);
    }

    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    final mq = MediaQuery.of(context).size;
    var scale = mq.aspectRatio * controller.value.aspectRatio;
    if (scale < 1) scale = 1 / scale;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Cover-scaled live preview (matches the overlay's cover mapping).
          ClipRect(
            child: Transform.scale(
              scale: scale,
              alignment: Alignment.center,
              child: Center(child: CameraPreview(controller)),
            ),
          ),
          CustomPaint(
            painter: TextOverlayPainter(
              regions: _display,
              frameWidth: _frameW,
              frameHeight: _frameH,
              revision: _revision,
            ),
          ),
          Align(
            alignment: Alignment.topCenter,
            child: HudBar(stats: _stats),
          ),
        ],
      ),
    );
  }
}

class _PermissionView extends StatelessWidget {
  const _PermissionView({required this.message, required this.onRetry});

  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 36),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.no_photography_outlined,
                  color: Colors.white54, size: 56),
              const SizedBox(height: 18),
              Text(
                message,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70, fontSize: 15),
              ),
              const SizedBox(height: 22),
              FilledButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: const Text('Try again'),
                style:
                    FilledButton.styleFrom(backgroundColor: GlyphColors.accent),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

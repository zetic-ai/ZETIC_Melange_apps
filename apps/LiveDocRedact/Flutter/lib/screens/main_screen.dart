import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/read_field.dart';
import '../services/ctc_decoder.dart';
import '../services/frame_data.dart';
import '../services/melange_service.dart';
import '../services/pii_classifier.dart';
import '../services/pipeline_isolate.dart';
import '../services/region_tracker.dart';
import '../theme.dart';
import '../widgets/hud_bar.dart';
import '../widgets/redaction_overlay.dart';
import '../widgets/stats_bar.dart';

/// Minimum recognizer confidence for a result to enter the cache.
const double kMinTextConfidence = 0.35;

/// Main screen: live camera feed, per-frame two-model pipeline, redaction
/// overlay and diagnostics HUD.
class MainScreen extends StatefulWidget {
  const MainScreen({
    super.key,
    required this.service,
    required this.pipeline,
    required this.decoder,
  });

  final MelangeService service;
  final DocPipeline pipeline;
  final CtcDecoder decoder;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  CameraDescription? _camera;

  bool _busy = false; // one frame in flight at a time; drop, never queue
  bool _streaming = false;
  String? _permissionError;

  final RegionTracker _tracker = RegionTracker();
  final PiiClassifier _classifier = PiiClassifier();

  int _budgetK = kRecognizerBudgetK;

  // UI snapshot (immutable per frame).
  List<RegionView> _regions = const [];
  Map<PiiClass, int> _piiCounts = const {};
  int _readCount = 0;
  int _trackedCount = 0;
  HudTimings _timings = const HudTimings();
  String _debugLine = 'waiting for first frame…';
  int _frameWidth = 0;
  int _frameHeight = 0;

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
            ? 'Camera permission denied. Enable it in Settings to use '
                'RedactLens.'
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
    if (_busy || !mounted) return;
    _busy = true;
    unawaited(_process(image));
  }

  Future<void> _process(CameraImage image) async {
    try {
      // iOS delivers the buffer already display-upright (measured on
      // PyroGuard: 720x1280 — do NOT assume landscape); Android delivers
      // sensor orientation and must be rotated. The HUD debug line shows the
      // real buffer WxH so this can be verified on-device.
      final int rotation = defaultTargetPlatform == TargetPlatform.iOS
          ? 0
          : (_camera?.sensorOrientation ?? 0);
      final frame =
          FrameData.fromCameraImage(image, rotationDegrees: rotation);

      final total = Stopwatch()..start();
      final sw = Stopwatch()..start();

      // 1. Detector preprocess (pipeline isolate; frame stays resident there).
      final det = await widget.pipeline.prepareFrame(frame);
      final preMs = sw.elapsedMilliseconds;

      // 2. Detector inference (main isolate — model handle bound here).
      sw.reset();
      final heatmap = widget.service.runDetector(det.input);
      final detMs = sw.elapsedMilliseconds;

      // 3. DB decode (isolate) + tracker update + budget scheduling.
      sw.reset();
      final db = await widget.pipeline.decodeHeatmap(heatmap);
      _tracker.update(db.regions);
      final scheduled = _tracker.scheduleRecognition(_budgetK);
      final decodeMs = sw.elapsedMilliseconds;

      // 4. Budgeted crops (isolate) + recognizer passes + CTC decode (main).
      sw.reset();
      var recMs = 0;
      if (scheduled.isNotEmpty) {
        final crops = await widget.pipeline.cropForRecognition(
            scheduled.map((r) => r.quad).toList(growable: false));
        for (var i = 0; i < scheduled.length; i++) {
          final logits = widget.service.runRecognizer(crops[i]);
          final rec = widget.decoder.decode(logits);
          // Cache even low-confidence/empty reads (as '') so the stagger
          // moves on; the re-read interval retries them later.
          _tracker.applyRecognition(
            scheduled[i].id,
            rec.confidence >= kMinTextConfidence ? rec.text : '',
            rec.confidence,
          );
        }
        recMs = sw.elapsedMilliseconds;
      }

      // 5. PII classification over ALL read fields (anchor adjacency can
      // re-classify neighbors as new texts arrive).
      final tracked = _tracker.tracked;
      final readFields = <TrackedRegion>[];
      final inputs = <PiiInputField>[];
      for (final r in tracked) {
        final text = r.text;
        if (text != null && text.isNotEmpty) {
          readFields.add(r);
          inputs.add(PiiInputField(
              bbox: r.bbox, text: text, confidence: r.textConfidence));
        }
      }
      final classified = _classifier.classify(inputs);
      for (var i = 0; i < readFields.length; i++) {
        readFields[i].piiClass = classified[i].piiClass;
      }

      total.stop();

      // 6. Immutable UI snapshot.
      final regions = <RegionView>[
        for (final r in tracked)
          RegionView(
            bbox: r.bbox,
            text: r.text,
            piiClass: r.piiClass,
            isRead: r.isRead,
          ),
      ];
      final counts = <PiiClass, int>{};
      var read = 0;
      for (final r in tracked) {
        if (r.isRead) read++;
        if (r.isRead && r.piiClass.isPii) {
          counts[r.piiClass] = (counts[r.piiClass] ?? 0) + 1;
        }
      }

      if (!mounted) return;
      setState(() {
        _regions = regions;
        _piiCounts = counts;
        _readCount = read;
        _trackedCount = tracked.length;
        _frameWidth = det.geometry.srcWidth;
        _frameHeight = det.geometry.srcHeight;
        _timings = HudTimings(
          preMs: preMs,
          detMs: detMs,
          decodeMs: decodeMs,
          recMs: recMs,
          cropsThisFrame: scheduled.length,
          totalMs: total.elapsedMilliseconds,
        );
        _debugLine = 'buf=${det.bufferWidth}x${det.bufferHeight} rot=$rotation '
            'upright=${det.geometry.srcWidth}x${det.geometry.srcHeight} '
            'heat[${db.stats}] regions=${db.regions.length} K=$_budgetK';
      });
    } catch (e) {
      if (mounted) {
        setState(() => _debugLine = 'pipeline error: $e');
      }
    } finally {
      _busy = false;
    }
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

  void _openSettings() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: RedactColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) {
        return StatefulBuilder(
          builder: (context, setSheet) {
            return Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Container(
                      width: 40,
                      height: 4,
                      decoration: BoxDecoration(
                        color: Colors.white24,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 18),
                  const Text(
                    'Pipeline settings',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Row(
                    children: [
                      const Text('Recognizer budget (crops/frame)',
                          style: TextStyle(color: Colors.white70)),
                      const Spacer(),
                      Text(
                        '$_budgetK',
                        style: const TextStyle(
                          color: RedactColors.accent,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                  Slider(
                    value: _budgetK.toDouble(),
                    min: 1,
                    max: 10,
                    divisions: 9,
                    label: '$_budgetK',
                    onChanged: (v) {
                      setSheet(() {});
                      setState(() => _budgetK = v.round());
                    },
                  ),
                  Text(
                    'Default $kRecognizerBudgetK for the CPU fallback '
                    '(~32 ms per crop). Raise once the console confirms '
                    'runtimeApType=NPU.',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.45),
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
            );
          },
        );
      },
    );
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
          // Cover-scaled live preview.
          ClipRect(
            child: Transform.scale(
              scale: scale,
              alignment: Alignment.center,
              child: Center(child: CameraPreview(controller)),
            ),
          ),
          if (kRedactionStyle == RedactionStyle.blur)
            RedactionBlurLayer(
              regions: _regions,
              frameWidth: _frameWidth,
              frameHeight: _frameHeight,
            ),
          CustomPaint(
            painter: RedactionOverlay(
              regions: _regions,
              frameWidth: _frameWidth,
              frameHeight: _frameHeight,
            ),
          ),
          Align(
            alignment: Alignment.topCenter,
            child: HudBar(
              timings: _timings,
              debugLine: _debugLine,
              onSettingsTap: _openSettings,
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: StatsBar(
              piiCounts: _piiCounts,
              readCount: _readCount,
              trackedCount: _trackedCount,
            ),
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
                style: FilledButton.styleFrom(
                    backgroundColor: RedactColors.accent),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud_bar.dart';
import '../widgets/stats_bar.dart';

/// Main screen: live camera feed with real-time PPE detection overlay + HUD.
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  CameraDescription? _camera;

  bool _busy = false; // one frame in flight at a time; extra frames dropped
  bool _streaming = false;
  bool _debugHud = false;
  String? _permissionError;

  List<Detection> _detections = const [];
  InferenceResult? _lastResult;

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
            ? 'Camera permission denied. Enable it in Settings to use SiteGuard.'
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
      // iOS delivers the buffer already display-upright (PyroGuard-verified);
      // Android delivers sensor orientation (usually 90) and must be rotated.
      // The debug HUD shows the real buffer WxH so this can be verified on
      // device rather than assumed.
      final int rotation = defaultTargetPlatform == TargetPlatform.iOS
          ? 0
          : (_camera?.sensorOrientation ?? 0);
      final result =
          await widget.service.detect(image, rotationDegrees: rotation);
      if (!mounted) return;
      setState(() {
        _detections = result.detections;
        _lastResult = result;
      });
    } catch (e) {
      // Dart-side failure only (native crashes never reach here). Show on the
      // HUD path instead of print — print does not surface in release.
      if (mounted) {
        setState(() => _detections = const []);
      }
      debugPrint('Inference error: $e');
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
    widget.service.dispose();
    super.dispose();
  }

  void _openSettings() {
    showModalBottomSheet<void>(
      context: context,
      backgroundColor: SiteColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) {
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
                'Detection thresholds',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              const Text(
                'Per-class thresholds are fixed to the Stage-0 validated '
                'values:\n'
                '  Hardhat 0.25 · Vest 0.15 · NO-Hardhat 0.15 · NO-Vest 0.15\n\n'
                'These were measured head-to-head on a labeled PPE test set '
                '(see model_selection.md); a live slider would silently change '
                'the validated operating point, so it is intentionally absent.',
                style: TextStyle(color: Colors.white70, fontSize: 13),
              ),
            ],
          ),
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

    final previewSize = controller.value.previewSize!;
    // previewSize is in sensor orientation (landscape): width is the long side.
    final int imgW = previewSize.width.toInt();
    final int imgH = previewSize.height.toInt();

    var hardhat = 0, vest = 0, noHardhat = 0, noVest = 0;
    for (final d in _detections) {
      switch (d.classId) {
        case kClassHardhat:
          hardhat++;
        case kClassVest:
          vest++;
        case kClassNoHardhat:
          noHardhat++;
        case kClassNoVest:
          noVest++;
      }
    }

    final result = _lastResult;

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
          // Detection boxes.
          CustomPaint(
            painter: DetectionOverlay(
              detections: _detections,
              imageWidth: imgW,
              imageHeight: imgH,
              sensorOrientation: _camera?.sensorOrientation ?? 90,
            ),
          ),
          // Top + bottom HUD.
          Align(
            alignment: Alignment.topCenter,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                HudBar(
                  preMs: result?.preMs ?? 0,
                  runMs: result?.runMs ?? 0,
                  postMs: result?.postMs ?? 0,
                  onSettingsTap: _openSettings,
                  onDebugTap: () => setState(() => _debugHud = !_debugHud),
                  debugOn: _debugHud,
                ),
                if (_debugHud && result != null) _DebugLine(result: result, detections: _detections),
              ],
            ),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: StatsBar(
              hardhatCount: hardhat,
              vestCount: vest,
              noHardhatCount: noHardhat,
              noVestCount: noVest,
            ),
          ),
        ],
      ),
    );
  }
}

/// On-screen diagnostics — the only observability that works in a release
/// device build (Dart print does not reach the native console there).
class _DebugLine extends StatelessWidget {
  const _DebugLine({required this.result, required this.detections});

  final InferenceResult result;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    final first = detections.isEmpty ? null : detections.first;
    final rawStr = first == null
        ? 'none'
        : '${first.label} ${first.confidence.toStringAsFixed(2)} '
            '(${first.rect.left.toStringAsFixed(2)},'
            '${first.rect.top.toStringAsFixed(2)},'
            '${first.rect.right.toStringAsFixed(2)},'
            '${first.rect.bottom.toStringAsFixed(2)})';
    return Container(
      margin: const EdgeInsets.fromLTRB(12, 6, 12, 0),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: SiteColors.scrim,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'buf=${result.bufWidth}x${result.bufHeight} '
        'rot=${result.rotationDegrees} '
        'src=${result.srcWidth}x${result.srcHeight} '
        'n=${detections.length} det0=$rawStr',
        style: const TextStyle(
          color: Colors.white70,
          fontSize: 10,
          fontFamily: 'monospace',
        ),
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
                    FilledButton.styleFrom(backgroundColor: SiteColors.accent),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

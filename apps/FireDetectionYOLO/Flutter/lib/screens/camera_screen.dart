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

/// Main screen: live camera feed with real-time detection overlay + HUD.
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

  bool _busy = false; // one frame in flight at a time
  bool _streaming = false;
  String? _permissionError;

  List<Detection> _detections = const [];
  int _latencyMs = 0;
  int _fireCount = 0;
  int _smokeCount = 0;

  double _confidence = 0.25; // default, slider range 0.1–0.8

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
            ? 'Camera permission denied. Enable it in Settings to use PyroGuard.'
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
      // iOS delivers the buffer already display-upright; Android delivers it in
      // sensor orientation (usually 90) and must be rotated so the model sees an
      // upright scene. After this, detections are upright on both platforms.
      final int rotation = defaultTargetPlatform == TargetPlatform.iOS
          ? 0
          : (_camera?.sensorOrientation ?? 0);
      final result =
          await widget.service.detect(image, _confidence, rotationDegrees: rotation);
      if (!mounted) return;
      var fire = 0;
      var smoke = 0;
      for (final d in result.detections) {
        if (d.isFire) {
          fire++;
        } else {
          smoke++;
        }
      }
      setState(() {
        _detections = result.detections;
        _latencyMs = result.latencyMs;
        _fireCount = fire;
        _smokeCount = smoke;
      });
    } catch (e) {
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
      backgroundColor: PyroColors.surface,
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
                    'Detection settings',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 20),
                  Row(
                    children: [
                      const Text('Confidence threshold',
                          style: TextStyle(color: Colors.white70)),
                      const Spacer(),
                      Text(
                        _confidence.toStringAsFixed(2),
                        style: const TextStyle(
                          color: PyroColors.accent,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                  Slider(
                    value: _confidence,
                    min: 0.1,
                    max: 0.8,
                    divisions: 70,
                    label: _confidence.toStringAsFixed(2),
                    onChanged: (v) {
                      setSheet(() {});
                      setState(() => _confidence = v);
                    },
                  ),
                  Text(
                    'Lower = more detections (and more false positives). '
                    'Default 0.25.',
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

    final previewSize = controller.value.previewSize!;
    // previewSize is in sensor orientation (landscape): width is the long side.
    final int imgW = previewSize.width.toInt();
    final int imgH = previewSize.height.toInt();

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
            child: HudBar(latencyMs: _latencyMs, onSettingsTap: _openSettings),
          ),
          Align(
            alignment: Alignment.bottomCenter,
            child: StatsBar(fireCount: _fireCount, smokeCount: _smokeCount),
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
                    FilledButton.styleFrom(backgroundColor: PyroColors.accent),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

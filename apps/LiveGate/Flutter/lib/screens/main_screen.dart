import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/gate_result.dart';
import '../models/geometry.dart';
import '../services/enrollment.dart';
import '../services/face_detector.dart';
import '../services/frame.dart';
import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/face_overlay.dart';

/// The live KYC gate: front-camera feed, per-frame liveness + match, verdict and
/// per-stage HUD, and an enroll action.
class MainScreen extends StatefulWidget {
  const MainScreen({
    super.key,
    required this.service,
    required this.enrollment,
  });

  final MelangeService service;
  final Enrollment enrollment;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  final FaceDetectorService _detector = FaceDetectorService();

  CameraController? _controller;
  CameraDescription? _camera;

  bool _busy = false;
  bool _streaming = false;
  bool _enrollPending = false;
  String? _permissionError;
  String? _toast;

  GateVerdict _verdict = const GateVerdict.noFace();
  int _prepMs = 0;
  int _padMs = 0;
  int _faceMs = 0;
  int _bufW = 0;
  int _bufH = 0;
  FaceBox? _lastBox;

  bool get _isFrontMirror =>
      _camera?.lensDirection == CameraLensDirection.front;

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
        (c) => c.lensDirection == CameraLensDirection.front,
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
            ? 'Camera permission denied. Enable it in Settings to use TrueFace.'
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

  /// Rotation applied to the raw buffer so the model + ML Kit share one upright
  /// space. iOS delivers the buffer already upright (0); Android delivers it in
  /// sensor orientation. The chosen value is surfaced on the HUD (buffer WxH) so
  /// it can be confirmed on-device — the reconciliation risk noted in HANDOFF.
  int get _rotation => defaultTargetPlatform == TargetPlatform.iOS
      ? 0
      : (_camera?.sensorOrientation ?? 0);

  Future<void> _process(CameraImage image) async {
    try {
      final detected = await _detector.detect(image, _rotation);
      if (!mounted) return;

      if (detected == null) {
        setState(() {
          _verdict = const GateVerdict.noFace();
          _lastBox = null;
        });
        return;
      }

      final frame = FrameData.fromCameraImage(image, rotationDegrees: _rotation);
      final req = PrepRequest(
        frame: frame,
        box: detected.box,
        landmarks: detected.landmarks,
      );

      if (_enrollPending) {
        await _runEnroll(req);
        return;
      }

      final result = await widget.service.process(
        req,
        enrolledNormalized: widget.enrollment.vector,
      );
      if (!mounted) return;
      setState(() {
        _verdict = result.verdict;
        _prepMs = result.prepMs;
        _padMs = result.padMs;
        _faceMs = result.faceMs;
        _bufW = result.bufferWidth;
        _bufH = result.bufferHeight;
        _lastBox = detected.box;
      });
    } catch (e) {
      // No Dart print on release device console; surface briefly on the HUD.
      if (mounted) setState(() => _toast = 'Inference error');
    } finally {
      _busy = false;
    }
  }

  Future<void> _runEnroll(PrepRequest req) async {
    // Only enroll a LIVE face. Run the pipeline first to check liveness.
    final result = await widget.service.process(req);
    if (!mounted) return;
    if (!result.verdict.live) {
      setState(() {
        _verdict = result.verdict;
        _toast = 'Hold a real, live face to enroll';
      });
      return;
    }
    final vec = await widget.service.embed(req);
    await widget.enrollment.enroll(vec);
    if (!mounted) return;
    setState(() {
      _enrollPending = false;
      _toast = 'Enrolled ✓ — reference saved on-device';
    });
  }

  Future<void> _clearEnrollment() async {
    await widget.enrollment.clear();
    if (!mounted) return;
    setState(() => _toast = 'Enrollment cleared');
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
    _detector.dispose();
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
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final mq = MediaQuery.of(context).size;
    var scale = mq.aspectRatio * controller.value.aspectRatio;
    if (scale < 1) scale = 1 / scale;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          ClipRect(
            child: Transform.scale(
              scale: scale,
              alignment: Alignment.center,
              child: Center(child: CameraPreview(controller)),
            ),
          ),
          CustomPaint(
            painter: FaceOverlay(
              box: _lastBox,
              bufferWidth: _bufW,
              bufferHeight: _bufH,
              decision: _verdict.decision,
              mirror: _isFrontMirror,
            ),
          ),
          SafeArea(
            child: Column(
              children: [
                _HudBar(
                  prepMs: _prepMs,
                  padMs: _padMs,
                  faceMs: _faceMs,
                  bufW: _bufW,
                  bufH: _bufH,
                ),
                const Spacer(),
                if (_toast != null) _Toast(text: _toast!),
                _VerdictPanel(
                  verdict: _verdict,
                  enrolled: widget.enrollment.isEnrolled,
                  onEnroll: () => setState(() {
                    _enrollPending = true;
                    _toast = 'Look at the camera to enroll…';
                  }),
                  onClear: _clearEnrollment,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Top HUD: per-stage latency, buffer WxH, and the on-device badge.
class _HudBar extends StatelessWidget {
  const _HudBar({
    required this.prepMs,
    required this.padMs,
    required this.faceMs,
    required this.bufW,
    required this.bufH,
  });

  final int prepMs;
  final int padMs;
  final int faceMs;
  final int bufW;
  final int bufH;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: GateColors.scrim,
        borderRadius: BorderRadius.circular(14),
      ),
      child: Row(
        children: [
          const Icon(Icons.lock_outline, size: 16, color: GateColors.accent),
          const SizedBox(width: 6),
          const Text('on-device · no cloud',
              style: TextStyle(color: Colors.white, fontSize: 12)),
          const Spacer(),
          Text(
            'prep ${prepMs}ms · pad ${padMs}ms · face ${faceMs}ms · buf ${bufW}x$bufH',
            style: const TextStyle(color: Colors.white70, fontSize: 11),
          ),
        ],
      ),
    );
  }
}

class _Toast extends StatelessWidget {
  const _Toast({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: GateColors.scrim,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(text,
          style: const TextStyle(color: Colors.white, fontSize: 13)),
    );
  }
}

/// Bottom panel: big verdict + match score badge + enroll/clear actions.
class _VerdictPanel extends StatelessWidget {
  const _VerdictPanel({
    required this.verdict,
    required this.enrolled,
    required this.onEnroll,
    required this.onClear,
  });

  final GateVerdict verdict;
  final bool enrolled;
  final VoidCallback onEnroll;
  final VoidCallback onClear;

  ({String label, Color color, IconData icon}) get _display {
    switch (verdict.decision) {
      case GateDecision.pass:
        return (label: 'LIVE · MATCH', color: GateColors.pass, icon: Icons.verified);
      case GateDecision.spoof:
        return (label: 'SPOOF', color: GateColors.fail, icon: Icons.block);
      case GateDecision.liveNoMatch:
        return (label: 'LIVE · NO MATCH', color: GateColors.warn, icon: Icons.person_off);
      case GateDecision.liveNoReference:
        return (label: 'LIVE · ENROLL FIRST', color: GateColors.warn, icon: Icons.how_to_reg);
      case GateDecision.noFace:
        return (label: 'NO FACE', color: Colors.white38, icon: Icons.face_retouching_off);
    }
  }

  @override
  Widget build(BuildContext context) {
    final d = _display;
    // Score is intentionally hidden for SPOOF and NO FACE (spec).
    final showScore = verdict.cosine != null &&
        (verdict.decision == GateDecision.pass ||
            verdict.decision == GateDecision.liveNoMatch);

    return Container(
      width: double.infinity,
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.fromLTRB(18, 16, 18, 18),
      decoration: BoxDecoration(
        color: GateColors.scrim,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: d.color.withValues(alpha: 0.6), width: 1.5),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(d.icon, color: d.color, size: 30),
              const SizedBox(width: 10),
              Text(
                d.label,
                style: TextStyle(
                  color: d.color,
                  fontSize: 26,
                  fontWeight: FontWeight.w900,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              _metric('liveness', verdict.liveScore.toStringAsFixed(3)),
              _metric('match', showScore ? verdict.cosine!.toStringAsFixed(3) : '—'),
            ],
          ),
          const SizedBox(height: 14),
          Row(
            children: [
              Expanded(
                child: FilledButton.icon(
                  onPressed: onEnroll,
                  icon: const Icon(Icons.how_to_reg, size: 18),
                  label: Text(enrolled ? 'Re-enroll' : 'Enroll'),
                  style:
                      FilledButton.styleFrom(backgroundColor: GateColors.accent),
                ),
              ),
              if (enrolled) ...[
                const SizedBox(width: 10),
                OutlinedButton.icon(
                  onPressed: onClear,
                  icon: const Icon(Icons.delete_outline, size: 18),
                  label: const Text('Clear'),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  Widget _metric(String label, String value) {
    return Column(
      children: [
        Text(label,
            style: const TextStyle(color: Colors.white54, fontSize: 11)),
        const SizedBox(height: 2),
        Text(value,
            style: const TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700)),
      ],
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
                style: FilledButton.styleFrom(backgroundColor: GateColors.accent),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

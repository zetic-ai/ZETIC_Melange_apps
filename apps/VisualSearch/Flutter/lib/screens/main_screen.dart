import 'dart:io';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../services/melange_service.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud_bar.dart';
import '../widgets/match_strip.dart';

/// The live demo: a viewfinder + a snap button. On snap we capture a still
/// (takePicture → JPEG, already oriented; deliberate deviation from a live
/// stream since SnapSeek is snap-then-search), run detect → crop → embed →
/// rank, and show the primary box, the embedded crop, per-stage latency, and
/// the top-K catalog matches.
class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

enum _Mode { live, searching, results }

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  CameraController? _camera;
  Future<void>? _cameraInit;
  bool _busy = false;

  _Mode _mode = _Mode.live;
  String? _capturedPath;
  SearchOutcome? _outcome;
  String? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      _camera = controller;
      _cameraInit = controller.initialize();
      await _cameraInit;
      if (mounted) setState(() {});
    } catch (e) {
      if (mounted) setState(() => _error = 'Camera error: $e');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final cam = _camera;
    if (cam == null || !cam.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      cam.dispose();
      _camera = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _snap() async {
    final cam = _camera;
    if (_busy || cam == null || !cam.value.isInitialized) return;
    setState(() {
      _busy = true;
      _mode = _Mode.searching;
      _error = null;
    });
    try {
      final XFile shot = await cam.takePicture();
      final Uint8List bytes = await shot.readAsBytes();
      final outcome = await widget.service.search(bytes);
      if (!mounted) return;
      setState(() {
        _capturedPath = shot.path;
        _outcome = outcome;
        _mode = _Mode.results;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _error = '$e';
        _mode = _Mode.live;
      });
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _reset() {
    setState(() {
      _mode = _Mode.live;
      _outcome = null;
      _capturedPath = null;
      _error = null;
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _camera?.dispose();
    widget.service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: SnapColors.bg,
      body: SafeArea(
        child: Column(
          children: [
            _header(),
            Expanded(
              child: _mode == _Mode.results
                  ? _resultsView()
                  : _liveView(),
            ),
          ],
        ),
      ),
    );
  }

  Widget _header() => Padding(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
        child: Row(
          children: [
            const Icon(Icons.center_focus_strong,
                color: SnapColors.accent, size: 22),
            const SizedBox(width: 8),
            RichText(
              text: const TextSpan(children: [
                TextSpan(
                    text: 'Snap',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w900)),
                TextSpan(
                    text: 'Seek',
                    style: TextStyle(
                        color: SnapColors.accent,
                        fontSize: 20,
                        fontWeight: FontWeight.w900)),
              ]),
            ),
            const Spacer(),
            Text(
              '${widget.service.gallery?.items.length ?? 0} in catalog',
              style: const TextStyle(color: SnapColors.textLo, fontSize: 12),
            ),
          ],
        ),
      );

  Widget _liveView() {
    final cam = _camera;
    if (_error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(_error!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.redAccent)),
        ),
      );
    }
    if (cam == null || !cam.value.isInitialized) {
      return const Center(
          child: CircularProgressIndicator(color: SnapColors.accent));
    }
    return Column(
      children: [
        Expanded(
          child: Stack(
            alignment: Alignment.center,
            children: [
              Center(child: CameraPreview(cam)),
              if (_mode == _Mode.searching)
                Container(
                  color: Colors.black45,
                  child: const Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        CircularProgressIndicator(color: SnapColors.accent),
                        SizedBox(height: 12),
                        Text('Searching…',
                            style: TextStyle(color: Colors.white)),
                      ],
                    ),
                  ),
                ),
              const IgnorePointer(child: _ReticleGuide()),
            ],
          ),
        ),
        _snapButton(),
      ],
    );
  }

  Widget _snapButton() => Padding(
        padding: const EdgeInsets.symmetric(vertical: 18),
        child: GestureDetector(
          onTap: _busy ? null : _snap,
          child: Container(
            width: 74,
            height: 74,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: _busy ? Colors.white24 : Colors.white,
              border: Border.all(color: SnapColors.accent, width: 4),
            ),
            child: const Icon(Icons.search, color: Colors.black, size: 30),
          ),
        ),
      );

  Widget _resultsView() {
    final outcome = _outcome!;
    final path = _capturedPath;
    return Column(
      children: [
        Expanded(
          child: Stack(
            children: [
              Center(
                child: AspectRatio(
                  aspectRatio: outcome.frameWidth / outcome.frameHeight,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      if (path != null)
                        Image.file(File(path), fit: BoxFit.fill)
                      else
                        const ColoredBox(color: Colors.black),
                      CustomPaint(
                        painter: DetectionOverlay(
                          primary: outcome.primary,
                          cropRect: outcome.cropRect,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              Positioned(
                  left: 0, right: 0, top: 0, child: HudBar(outcome: outcome)),
            ],
          ),
        ),
        MatchStrip(results: outcome.results),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          child: SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _reset,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Snap again'),
            ),
          ),
        ),
      ],
    );
  }
}

/// A subtle center reticle to guide the user to frame the product.
class _ReticleGuide extends StatelessWidget {
  const _ReticleGuide();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Container(
        width: 220,
        height: 220,
        decoration: BoxDecoration(
          border: Border.all(color: SnapColors.accent.withValues(alpha: 0.5)),
          borderRadius: BorderRadius.circular(16),
        ),
      ),
    );
  }
}

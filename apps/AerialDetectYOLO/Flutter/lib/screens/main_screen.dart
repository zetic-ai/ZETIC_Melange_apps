import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../main.dart';
import '../models/detection.dart';
import '../services/melange_service.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud.dart';
import '../widgets/photo_detect_view.dart';

/// Which detection source is active.
enum DemoMode { live, photo }

/// The demo: a Live/Photo toggle over the shared Melange pipeline. Live streams
/// the camera; Photo runs the SAME model on a picked still (SkyScout is trained
/// on aerial imagery, so a real drone photo is where it actually fires).
class MainScreen extends StatefulWidget {
  const MainScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> with WidgetsBindingObserver {
  CameraController? _controller;
  bool _streaming = false;

  List<Detection> _detections = const <Detection>[];
  FrameTimings? _timings;
  Size _bufferSize = Size.zero;
  int _sensorOrientation = 0;

  DemoMode _mode = DemoMode.live;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    if (gCameras.isEmpty) return;
    final CameraDescription cam = gCameras.firstWhere(
      (CameraDescription c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => gCameras.first,
    );
    _sensorOrientation = cam.sensorOrientation;

    final CameraController controller = CameraController(
      cam,
      ResolutionPreset.high,
      enableAudio: false,
      // Cheapest usable format per platform.
      imageFormatGroup: Platform.isIOS
          ? ImageFormatGroup.bgra8888
          : ImageFormatGroup.yuv420,
    );
    _controller = controller;
    await controller.initialize();
    if (!mounted) return;

    await controller.startImageStream(_onFrame);
    _streaming = true;
    setState(() {});
  }

  Future<void> _setMode(DemoMode mode) async {
    if (mode == _mode) return;
    final CameraController? c = _controller;
    if (mode == DemoMode.photo) {
      // Free the model for the still path: stop streaming camera frames.
      if (c != null && _streaming) {
        await c.stopImageStream();
        _streaming = false;
      }
    } else {
      if (c != null && c.value.isInitialized && !_streaming) {
        await c.startImageStream(_onFrame);
        _streaming = true;
      }
    }
    if (!mounted) return;
    setState(() {
      _mode = mode;
      // Clear live results so the HUD doesn't show stale live counts in photo.
      if (mode == DemoMode.photo) {
        _detections = const <Detection>[];
        _timings = null;
      }
    });
  }

  void _onFrame(CameraImage image) {
    // _busy frame-guard lives in the service; drop frames rather than queue.
    if (_mode != DemoMode.live) return;
    if (widget.service.busy) return;
    if (_bufferSize == Size.zero) {
      _bufferSize = Size(image.width.toDouble(), image.height.toDouble());
    }
    widget.service.infer(image).then((result) {
      if (!mounted || result == null) return;
      setState(() {
        _detections = result.detections;
        _timings = result.timings;
      });
    });
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? c = _controller;
    if (c == null || !c.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      if (_streaming) {
        c.stopImageStream();
        _streaming = false;
      }
    } else if (state == AppLifecycleState.resumed &&
        !_streaming &&
        _mode == DemoMode.live) {
      c.startImageStream(_onFrame);
      _streaming = true;
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    widget.service.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: <Widget>[
          _mode == DemoMode.live
              ? _buildLive()
              : PhotoDetectView(service: widget.service),
          SafeArea(
            child: Align(
              alignment: Alignment.topCenter,
              child: Padding(
                padding: const EdgeInsets.only(top: 8),
                child: _modeToggle(),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _modeToggle() {
    return SegmentedButton<DemoMode>(
      segments: const <ButtonSegment<DemoMode>>[
        ButtonSegment<DemoMode>(
          value: DemoMode.live,
          label: Text('Live'),
          icon: Icon(Icons.videocam),
        ),
        ButtonSegment<DemoMode>(
          value: DemoMode.photo,
          label: Text('Photo'),
          icon: Icon(Icons.photo),
        ),
      ],
      selected: <DemoMode>{_mode},
      onSelectionChanged: (Set<DemoMode> s) => _setMode(s.first),
      showSelectedIcon: false,
    );
  }

  Widget _buildLive() {
    final CameraController? c = _controller;
    if (c == null || !c.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    // Overlay image size: the buffer the model saw, in its native orientation.
    final Size imageSize = _bufferSize == Size.zero
        ? Size(c.value.previewSize?.height ?? 1, c.value.previewSize?.width ?? 1)
        : _bufferSize;

    return Stack(
      fit: StackFit.expand,
      children: <Widget>[
        FittedBox(
          fit: BoxFit.cover,
          child: SizedBox(
            width: c.value.previewSize?.height ?? 1,
            height: c.value.previewSize?.width ?? 1,
            child: CameraPreview(c),
          ),
        ),
        DetectionOverlay(detections: _detections, imageSize: imageSize),
        Hud(
          detections: _detections,
          timings: _timings,
          bufferSize: imageSize,
          sensorOrientation: _sensorOrientation,
        ),
      ],
    );
  }
}

import 'dart:io' show Platform;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/frame_data.dart';
import '../services/melange_service.dart';
import '../services/plate_ocr.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud.dart';

/// Live camera + per-frame detection. Streams the cheapest usable pixel format,
/// feeds each frame to the dedicated inference isolate (dropping frames while
/// busy), and overlays plate boxes + a HUD.
class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen>
    with WidgetsBindingObserver {
  CameraController? _controller;
  bool _streaming = false;

  InferenceResult _result = InferenceResult.empty;
  int _bufW = 0;
  int _bufH = 0;
  int _rotation = 0;

  // --- On-device plate OCR (iOS Apple Vision) state ---
  // OCR runs on the MAIN isolate (platform channels are root-isolate only),
  // never on the inference isolate. It is throttled and single-flighted so it
  // can never block or pile up behind the camera stream.
  static const Duration _ocrThrottle = Duration(milliseconds: 350);
  bool _ocrInFlight = false;
  DateTime _lastOcrAt = DateTime.fromMillisecondsSinceEpoch(0);
  // Recognized text cached per tracked box (quantized center key) so the
  // overlay stays stable between OCR passes. `_ocrRevision` forces the overlay
  // to repaint when the cache changes even though the detection list is the
  // same instance.
  final Map<int, String> _plateText = {};
  String? _latestPlate;
  int _ocrRevision = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initCamera();
  }

  Future<void> _initCamera() async {
    final cameras = await availableCameras();
    final back = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );
    final controller = CameraController(
      back,
      ResolutionPreset.medium,
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
    // iOS BGRA buffer arrives upright -> no rotation. Android YUV420 is
    // sensor-landscape -> rotate by sensorOrientation (Tier C device calibration).
    _rotation = defaultTargetPlatform == TargetPlatform.iOS
        ? 0
        : back.sensorOrientation;
    setState(() => _controller = controller);
    await controller.startImageStream(_onFrame);
    _streaming = true;
  }

  void _onFrame(CameraImage image) {
    if (widget.service.isBusy) return; // drop, don't queue
    final frame = _toFrameData(image);
    if (frame == null) return;
    _bufW = image.width;
    _bufH = image.height;
    // Capture the BGRA bytes for a possible OCR crop of THIS frame. The camera
    // plugin hands us a Dart-owned Uint8List, so it stays valid in the async
    // callback below. Only the iOS BGRA path supports OCR (see _maybeRunOcr).
    final isBgra = image.format.group == ImageFormatGroup.bgra8888;
    final Uint8List? ocrBytes = isBgra ? image.planes.first.bytes : null;
    final int ocrBpr = isBgra ? image.planes.first.bytesPerRow : 0;
    widget.service.detect(frame).then((r) {
      if (r == null || !mounted) return;
      setState(() => _result = r);
      if (ocrBytes != null) {
        _maybeRunOcr(r, ocrBytes, ocrBpr, image.width, image.height);
      }
    });
  }

  /// Run Apple Vision OCR on the highest-confidence plate of [r], throttled and
  /// single-flighted so the camera stays smooth. iOS-only; assumes the BGRA
  /// upright buffer (rotation 0) so detection coords map straight to the crop.
  void _maybeRunOcr(
    InferenceResult r,
    Uint8List bgra,
    int bytesPerRow,
    int width,
    int height,
  ) {
    if (!Platform.isIOS) return; // Android: Vision unavailable — skip cleanly.
    if (_rotation != 0) return; // crop assumes upright == buffer space.
    if (_ocrInFlight || r.detections.isEmpty) return;
    if (DateTime.now().difference(_lastOcrAt) < _ocrThrottle) return;

    var best = r.detections.first;
    for (final d in r.detections) {
      if (d.confidence > best.confidence) best = d;
    }

    _ocrInFlight = true;
    _lastOcrAt = DateTime.now();
    PlateOcr.recognize(
      bgra: bgra,
      bytesPerRow: bytesPerRow,
      width: width,
      height: height,
      left: best.left,
      top: best.top,
      right: best.right,
      bottom: best.bottom,
    ).then((text) {
      _ocrInFlight = false;
      if (!mounted || text == null) return;
      setState(() {
        _plateText[_boxKey(best)] = text;
        _latestPlate = text;
        _ocrRevision++;
        _trimPlateCache();
      });
    }, onError: (_) {
      _ocrInFlight = false;
    });
  }

  /// Quantize a box center to a 32px grid so a moving plate keeps the same cache
  /// key across frames (stable overlay text).
  int _boxKey(Detection d) {
    final cx = ((d.left + d.right) / 2 / 32).round();
    final cy = ((d.top + d.bottom) / 2 / 32).round();
    return cy * 100000 + cx;
  }

  String? _plateFor(Detection d) => _plateText[_boxKey(d)];

  void _trimPlateCache() {
    while (_plateText.length > 16) {
      _plateText.remove(_plateText.keys.first);
    }
  }

  FrameData? _toFrameData(CameraImage image) {
    switch (image.format.group) {
      case ImageFormatGroup.bgra8888:
        final p = image.planes.first;
        return FrameData(
          format: FramePixelFormat.bgra8888,
          width: image.width,
          height: image.height,
          rotationDegrees: _rotation,
          plane0: p.bytes,
          bytesPerRow0: p.bytesPerRow,
        );
      case ImageFormatGroup.yuv420:
        if (image.planes.length < 3) return null;
        final y = image.planes[0];
        final u = image.planes[1];
        final v = image.planes[2];
        return FrameData(
          format: FramePixelFormat.yuv420,
          width: image.width,
          height: image.height,
          rotationDegrees: _rotation,
          plane0: y.bytes,
          bytesPerRow0: y.bytesPerRow,
          plane1: u.bytes,
          plane2: v.bytes,
          bytesPerRow1: u.bytesPerRow,
          bytesPerRow2: v.bytesPerRow,
          pixelStride1: u.bytesPerPixel ?? 1,
          pixelStride2: v.bytesPerPixel ?? 1,
        );
      default:
        return null;
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final c = _controller;
    if (c == null || !c.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      if (_streaming) {
        c.stopImageStream();
        _streaming = false;
      }
    } else if (state == AppLifecycleState.resumed && !_streaming) {
      c.startImageStream(_onFrame);
      _streaming = true;
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    widget.service.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) {
      return const Scaffold(
        backgroundColor: AppTheme.bg,
        body: Center(
          child: CircularProgressIndicator(color: AppTheme.accent),
        ),
      );
    }

    final imageW = _result.imageWidth > 0 ? _result.imageWidth : _bufH;
    final imageH = _result.imageHeight > 0 ? _result.imageHeight : _bufW;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Fill the screen with the preview (cover) and overlay boxes mapped
          // through the same cover transform.
          LayoutBuilder(
            builder: (context, constraints) {
              return ClipRect(
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    FittedBox(
                      fit: BoxFit.cover,
                      child: SizedBox(
                        width: controller.value.previewSize?.height ?? imageW.toDouble(),
                        height: controller.value.previewSize?.width ?? imageH.toDouble(),
                        child: CameraPreview(controller),
                      ),
                    ),
                    DetectionOverlay(
                      detections: _result.detections,
                      imageWidth: imageW,
                      imageHeight: imageH,
                      plateOf: _plateFor,
                      revision: _ocrRevision,
                    ),
                  ],
                ),
              );
            },
          ),
          SafeArea(
            child: Hud(
              plateCount: _result.detections.length,
              latencyMs: _result.latencyMs,
              bufWidth: _bufW,
              bufHeight: _bufH,
              rotation: _rotation,
              imageWidth: imageW,
              imageHeight: imageH,
              plateText: _latestPlate,
            ),
          ),
        ],
      ),
    );
  }
}

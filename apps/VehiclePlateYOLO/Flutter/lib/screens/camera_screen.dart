import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../services/frame_data.dart';
import '../services/melange_service.dart';
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
    widget.service.detect(frame).then((r) {
      if (r == null || !mounted) return;
      setState(() => _result = r);
    });
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
            ),
          ),
        ],
      ),
    );
  }
}

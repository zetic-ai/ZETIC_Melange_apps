import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';

import '../models/detection.dart';
import '../services/melange_service.dart';
import '../services/plate_ocr.dart';
import '../services/still_image.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud.dart';

/// Asset path of the bundled demo still used by the "Try sample" button.
const String _kSampleAsset = 'assets/samples/plate_sample.jpg';

/// Photo mode: pick a still from the library (or load the bundled sample), run
/// the SAME detection pipeline ([MelangeService.detect]) on it, then run the
/// SAME Apple Vision OCR ([PlateOcr]) on EACH detected plate. Boxes + per-plate
/// text are overlaid on the still with BoxFit.contain math. The still is decoded
/// to the identical upright BGRA [FrameData] the live camera produces, so
/// nothing in the model path is duplicated — only the pixel source (gallery
/// pick vs bundled asset vs camera stream) differs.
class PhotoScreen extends StatefulWidget {
  const PhotoScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<PhotoScreen> createState() => _PhotoScreenState();
}

class _PhotoScreenState extends State<PhotoScreen> {
  final ImagePicker _picker = ImagePicker();

  Widget? _display; // how the source image is shown (Image.file / Image.asset)
  int _imageW = 0;
  int _imageH = 0;
  List<Detection> _detections = const [];
  final Map<Detection, String> _plates = {}; // identity-keyed per run
  String? _latestPlate;
  double _latencyMs = 0;
  bool _busy = false;
  String? _error;

  /// Pick a still from the gallery and analyze it.
  Future<void> _pick() async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final picked = await _picker.pickImage(source: ImageSource.gallery);
      if (picked == null) {
        if (mounted) setState(() => _busy = false);
        return;
      }
      final bytes = await picked.readAsBytes();
      await _analyze(bytes, Image.file(File(picked.path), fit: BoxFit.contain));
    } catch (e) {
      _fail(e);
    }
  }

  /// Load the bundled demo image and run the SAME detect+OCR path as [_pick].
  Future<void> _trySample() async {
    if (_busy) return;
    setState(() {
      _busy = true;
      _error = null;
    });
    try {
      final data = await rootBundle.load(_kSampleAsset);
      final bytes = data.buffer.asUint8List();
      await _analyze(bytes, const Image(
        image: AssetImage(_kSampleAsset),
        fit: BoxFit.contain,
      ));
    } catch (e) {
      _fail(e);
    }
  }

  /// Shared still-image path: decode off the UI isolate, run the SAME detection
  /// pipeline as the live camera, then OCR EVERY detected plate (one Vision pass
  /// each) so each box gets its own recognized-text label.
  Future<void> _analyze(Uint8List bytes, Widget display) async {
    // Decode + EXIF-bake + downscale off the UI isolate.
    final still = await compute(decodeStillImage, bytes);
    if (still == null) {
      if (mounted) {
        setState(() {
          _busy = false;
          _error = 'Could not decode that image.';
        });
      }
      return;
    }

    // SAME detection pipeline as the live camera.
    final result = await widget.service.detect(still.toFrameData());
    final dets = result?.detections ?? const <Detection>[];

    // SAME Apple Vision OCR as the live camera, one pass PER plate. The still is
    // upright (rotation 0), so no rotation gating is needed. Every detection is
    // OCR'd and labeled independently (not just the top-confidence one).
    final plates = <Detection, String>{};
    String? latest;
    for (final d in dets) {
      final text = await PlateOcr.recognize(
        bgra: still.bgra,
        bytesPerRow: still.bytesPerRow,
        width: still.width,
        height: still.height,
        left: d.left,
        top: d.top,
        right: d.right,
        bottom: d.bottom,
      );
      if (text != null) {
        plates[d] = text;
        latest = text;
      }
    }

    if (!mounted) return;
    setState(() {
      _display = display;
      _imageW = still.width;
      _imageH = still.height;
      _detections = dets;
      _plates
        ..clear()
        ..addAll(plates);
      _latestPlate = latest;
      _latencyMs = result?.latencyMs ?? 0;
      _busy = false;
    });
  }

  void _fail(Object e) {
    if (!mounted) return;
    setState(() {
      _busy = false;
      _error = '$e';
    });
  }

  String? _plateFor(Detection d) => _plates[d];

  @override
  Widget build(BuildContext context) {
    final display = _display;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (display == null)
            _EmptyState(busy: _busy)
          else
            LayoutBuilder(
              builder: (context, constraints) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    display,
                    DetectionOverlay(
                      detections: _detections,
                      imageWidth: _imageW,
                      imageHeight: _imageH,
                      plateOf: _plateFor,
                      cover: false,
                    ),
                  ],
                );
              },
            ),
          if (display != null)
            SafeArea(
              child: Hud(
                plateCount: _detections.length,
                latencyMs: _latencyMs,
                bufWidth: _imageW,
                bufHeight: _imageH,
                rotation: 0,
                imageWidth: _imageW,
                imageHeight: _imageH,
                plateText: _latestPlate,
              ),
            ),
          if (_error != null)
            Positioned(
              left: 16,
              right: 16,
              top: MediaQuery.of(context).padding.top + 80,
              child: _ErrorBanner(_error!),
            ),
          // Action buttons, kept clear of the mode toggle at the very bottom.
          Positioned(
            left: 0,
            right: 0,
            bottom: 92,
            child: Center(
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  OutlinedButton.icon(
                    onPressed: _busy ? null : _trySample,
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppTheme.accent,
                      side: const BorderSide(color: AppTheme.accent),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 18,
                        vertical: 14,
                      ),
                    ),
                    icon: const Icon(Icons.auto_awesome_outlined),
                    label: const Text('Try sample'),
                  ),
                  const SizedBox(width: 12),
                  FilledButton.icon(
                    onPressed: _busy ? null : _pick,
                    style: FilledButton.styleFrom(
                      backgroundColor: AppTheme.accent,
                      foregroundColor: AppTheme.bg,
                      padding: const EdgeInsets.symmetric(
                        horizontal: 22,
                        vertical: 14,
                      ),
                    ),
                    icon: _busy
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: AppTheme.bg,
                            ),
                          )
                        : const Icon(Icons.photo_library_outlined),
                    label: Text(_busy
                        ? 'Analyzing…'
                        : (display == null ? 'Pick a photo' : 'Pick another')),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({required this.busy});

  final bool busy;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.image_search_outlined,
            size: 72,
            color: AppTheme.accent.withValues(alpha: busy ? 0.4 : 0.8),
          ),
          const SizedBox(height: 16),
          Text(
            busy ? 'Analyzing…' : 'Pick a photo to detect plates',
            style: const TextStyle(color: AppTheme.textMuted),
          ),
        ],
      ),
    );
  }
}

class _ErrorBanner extends StatelessWidget {
  const _ErrorBanner(this.message);

  final String message;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.warn.withValues(alpha: 0.5)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: AppTheme.warn, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              message,
              style: const TextStyle(color: AppTheme.textPrimary, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}

import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/detection.dart';
import '../services/melange_service.dart';
import '../services/plate_ocr.dart';
import '../services/still_image.dart';
import '../theme.dart';
import '../widgets/detection_overlay.dart';
import '../widgets/hud.dart';

/// Photo mode: pick a still from the library, run the SAME detection pipeline
/// ([MelangeService.detect]) on it, then run the SAME Apple Vision OCR
/// ([PlateOcr]) on each detected plate. Boxes + plate text are overlaid on the
/// still with BoxFit.contain math. The still is decoded to the identical
/// upright BGRA [FrameData] the live camera produces, so nothing in the model
/// path is duplicated — only the pixel source (gallery vs camera stream) differs.
class PhotoScreen extends StatefulWidget {
  const PhotoScreen({super.key, required this.service});

  final MelangeService service;

  @override
  State<PhotoScreen> createState() => _PhotoScreenState();
}

class _PhotoScreenState extends State<PhotoScreen> {
  final ImagePicker _picker = ImagePicker();

  XFile? _file; // displayed via Image.file (EXIF-applied == baked orientation)
  int _imageW = 0;
  int _imageH = 0;
  List<Detection> _detections = const [];
  final Map<Detection, String> _plates = {}; // identity-keyed per pick
  String? _latestPlate;
  double _latencyMs = 0;
  bool _busy = false;
  String? _error;

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

      // SAME Apple Vision OCR as the live camera, one pass per plate. The still
      // is upright (rotation 0), so no rotation gating is needed.
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
        _file = picked;
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
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _busy = false;
        _error = '$e';
      });
    }
  }

  String? _plateFor(Detection d) => _plates[d];

  @override
  Widget build(BuildContext context) {
    final file = _file;
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (file == null)
            _EmptyState(busy: _busy)
          else
            LayoutBuilder(
              builder: (context, constraints) {
                return Stack(
                  fit: StackFit.expand,
                  children: [
                    Image.file(File(file.path), fit: BoxFit.contain),
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
          if (file != null)
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
          // Pick button, kept clear of the mode toggle at the very bottom.
          Positioned(
            left: 0,
            right: 0,
            bottom: 92,
            child: Center(
              child: FilledButton.icon(
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
                    : (file == null ? 'Pick a photo' : 'Pick another')),
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

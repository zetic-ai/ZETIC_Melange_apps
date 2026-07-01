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

/// Upload-only plate reader. The user picks a still from the library; the SAME
/// detection pipeline ([MelangeService.detect]) runs on it, then the SAME Apple
/// Vision OCR ([PlateOcr]) runs on EACH detected plate. Results are shown as a
/// list where every row pairs a ZOOMED CROP of the plate region with its OCR'd
/// text, so the plate can be read and compared to what OCR returned. There is no
/// live camera and no "try sample" — just upload → read → verify.
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
  final Map<Detection, Uint8List> _crops = {}; // zoomed plate crops (PNG)
  double _latencyMs = 0;
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    // This screen now owns the shared inference service (no HomeScreen wrapper).
    widget.service.close();
    super.dispose();
  }

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

  /// Still-image path: decode off the UI isolate, run the SAME detection pipeline
  /// as before, OCR EVERY detected plate (one Vision pass each), and encode a
  /// zoomed crop of each plate region for the readable results list.
  Future<void> _analyze(Uint8List bytes, Widget display) async {
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

    final result = await widget.service.detect(still.toFrameData());
    final dets = result?.detections ?? const <Detection>[];

    // Per plate: OCR (iOS Vision; null on Android) + a zoomed readable crop.
    final plates = <Detection, String>{};
    final crops = <Detection, Uint8List>{};
    for (final d in dets) {
      final crop = encodePlateCropPng(
        still,
        d.left,
        d.top,
        d.right,
        d.bottom,
      );
      if (crop != null) crops[d] = crop;

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
      if (text != null) plates[d] = text;
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
      _crops
        ..clear()
        ..addAll(crops);
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

  @override
  Widget build(BuildContext context) {
    final display = _display;
    final hasResult = display != null;
    return Scaffold(
      backgroundColor: AppTheme.bg,
      body: SafeArea(
        child: Column(
          children: [
            _TopBar(plateCount: _detections.length, hasResult: hasResult),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 12),
                child: hasResult
                    ? _ImageWithBoxes(
                        display: display,
                        imageW: _imageW,
                        imageH: _imageH,
                        detections: _detections,
                      )
                    : _EmptyState(busy: _busy),
              ),
            ),
            if (_error != null)
              Padding(
                padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
                child: _ErrorBanner(_error!),
              ),
            if (hasResult)
              _ResultsList(
                detections: _detections,
                crops: _crops,
                plates: _plates,
              ),
            _ActionBar(
              busy: _busy,
              hasResult: hasResult,
              onUpload: _pick,
            ),
            _Footer(latencyMs: _latencyMs),
          ],
        ),
      ),
    );
  }
}

/// Compact top bar: app name + plate count. Nothing overlaps the image.
class _TopBar extends StatelessWidget {
  const _TopBar({required this.plateCount, required this.hasResult});

  final int plateCount;
  final bool hasResult;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 10, 16, 10),
      child: Row(
        children: [
          const Icon(Icons.directions_car_filled,
              color: AppTheme.accent, size: 22),
          const SizedBox(width: 10),
          const Text(
            'PlateHawk',
            style: TextStyle(
              color: AppTheme.textPrimary,
              fontSize: 18,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.3,
            ),
          ),
          const Spacer(),
          if (hasResult)
            Container(
              padding:
                  const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppTheme.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: AppTheme.accentSoft),
              ),
              child: Text(
                plateCount == 1 ? '1 plate' : '$plateCount plates',
                style: const TextStyle(
                  color: AppTheme.accent,
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// The uploaded image (BoxFit.contain) with thin numbered boxes on top.
class _ImageWithBoxes extends StatelessWidget {
  const _ImageWithBoxes({
    required this.display,
    required this.imageW,
    required this.imageH,
    required this.detections,
  });

  final Widget display;
  final int imageW;
  final int imageH;
  final List<Detection> detections;

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Stack(
        fit: StackFit.expand,
        children: [
          Positioned.fill(child: display),
          DetectionOverlay(
            detections: detections,
            imageWidth: imageW,
            imageHeight: imageH,
          ),
        ],
      ),
    );
  }
}

/// Scrollable results: each row = zoomed plate crop NEXT TO its OCR'd text, so
/// the user can read the plate and check the OCR against it. This is the core
/// "verifiable accuracy" surface.
class _ResultsList extends StatelessWidget {
  const _ResultsList({
    required this.detections,
    required this.crops,
    required this.plates,
  });

  final List<Detection> detections;
  final Map<Detection, Uint8List> crops;
  final Map<Detection, String> plates;

  @override
  Widget build(BuildContext context) {
    if (detections.isEmpty) {
      return const Padding(
        padding: EdgeInsets.symmetric(vertical: 16),
        child: Text(
          'No plates found in this photo.',
          style: TextStyle(color: AppTheme.textMuted),
        ),
      );
    }
    return Container(
      constraints: const BoxConstraints(maxHeight: 210),
      margin: const EdgeInsets.only(top: 8),
      child: ListView.separated(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
        shrinkWrap: true,
        itemCount: detections.length,
        separatorBuilder: (context, index) => const SizedBox(height: 8),
        itemBuilder: (context, i) {
          final d = detections[i];
          return _PlateRow(
            index: i + 1,
            crop: crops[d],
            text: plates[d],
            confidence: d.confidence,
          );
        },
      ),
    );
  }
}

class _PlateRow extends StatelessWidget {
  const _PlateRow({
    required this.index,
    required this.crop,
    required this.text,
    required this.confidence,
  });

  final int index;
  final Uint8List? crop;
  final String? text;
  final double confidence;

  @override
  Widget build(BuildContext context) {
    final hasText = text != null && text!.isNotEmpty;
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: AppTheme.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.accentSoft),
      ),
      child: Row(
        children: [
          // Index badge matching the box on the image.
          Container(
            width: 24,
            height: 24,
            alignment: Alignment.center,
            decoration: const BoxDecoration(
              color: AppTheme.accent,
              shape: BoxShape.circle,
            ),
            child: Text(
              '$index',
              style: const TextStyle(
                color: AppTheme.bg,
                fontSize: 12,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          const SizedBox(width: 10),
          // Zoomed crop of the plate — the readable proof.
          Container(
            width: 132,
            height: 60,
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppTheme.accentSoft),
            ),
            clipBehavior: Clip.antiAlias,
            child: crop != null
                ? Image.memory(
                    crop!,
                    fit: BoxFit.contain,
                    filterQuality: FilterQuality.medium,
                    gaplessPlayback: true,
                  )
                : const Center(
                    child: Icon(Icons.image_not_supported_outlined,
                        color: AppTheme.textMuted, size: 20),
                  ),
          ),
          const SizedBox(width: 12),
          // OCR text (or a graceful note where Vision isn't available).
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  hasText
                      ? text!
                      : (Platform.isIOS ? 'No text read' : 'OCR: iOS only'),
                  style: TextStyle(
                    color: hasText ? AppTheme.textPrimary : AppTheme.textMuted,
                    fontSize: hasText ? 22 : 14,
                    fontWeight: FontWeight.w800,
                    letterSpacing: hasText ? 1.5 : 0,
                    fontFeatures: const [FontFeature.tabularFigures()],
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'detected ${(confidence * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(
                    color: AppTheme.textMuted,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// Bottom action: the single primary "Upload photo" / "Pick another" button.
class _ActionBar extends StatelessWidget {
  const _ActionBar({
    required this.busy,
    required this.hasResult,
    required this.onUpload,
  });

  final bool busy;
  final bool hasResult;
  final VoidCallback onUpload;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 4),
      child: SizedBox(
        width: double.infinity,
        child: FilledButton.icon(
          onPressed: busy ? null : onUpload,
          style: FilledButton.styleFrom(
            backgroundColor: AppTheme.accent,
            foregroundColor: AppTheme.bg,
            padding: const EdgeInsets.symmetric(vertical: 16),
          ),
          icon: busy
              ? const SizedBox(
                  width: 18,
                  height: 18,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: AppTheme.bg,
                  ),
                )
              : const Icon(Icons.upload_outlined),
          label: Text(
            busy
                ? 'Analyzing…'
                : (hasResult ? 'Pick another' : 'Upload photo'),
            style: const TextStyle(fontWeight: FontWeight.w700),
          ),
        ),
      ),
    );
  }
}

class _Footer extends StatelessWidget {
  const _Footer({required this.latencyMs});

  final double latencyMs;

  @override
  Widget build(BuildContext context) {
    final ms = latencyMs > 0 ? ' • ${latencyMs.toStringAsFixed(0)} ms' : '';
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Text(
        'on-device$ms',
        style: const TextStyle(color: AppTheme.textMuted, fontSize: 10),
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
            busy ? 'Analyzing…' : 'Upload a photo to detect & read plates',
            textAlign: TextAlign.center,
            style: const TextStyle(color: AppTheme.textMuted),
          ),
          const SizedBox(height: 6),
          const Text(
            'Each plate is shown zoomed next to its reading,\n'
            'so you can check the result yourself.',
            textAlign: TextAlign.center,
            style: TextStyle(color: AppTheme.textMuted, fontSize: 12),
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

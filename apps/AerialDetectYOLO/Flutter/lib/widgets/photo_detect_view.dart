import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show ByteData, Uint8List, rootBundle;
import 'package:image_picker/image_picker.dart';

import '../models/detection.dart';
import '../services/image_decoder.dart';
import '../services/melange_service.dart';
import 'hud.dart';
import 'photo_overlay.dart';

/// Photo-upload detection mode.
///
/// The user picks an aerial/drone photo from the library; it is decoded (EXIF
/// baked) off the UI isolate, run through the SAME Melange pipeline as the live
/// feed via [MelangeService.inferStill], and the boxes are drawn over the still
/// with a per-class-count HUD. SkyScout is trained on aerial imagery, so this is
/// where it actually fires (ground-level live scenes stay empty).
class PhotoDetectView extends StatefulWidget {
  const PhotoDetectView({super.key, required this.service});

  final MelangeService service;

  @override
  State<PhotoDetectView> createState() => _PhotoDetectViewState();
}

/// Bundled demo aerial image (a drone car-park shot) shipped with the app so the
/// detector can be exercised without a photo library — see pubspec `assets`.
const String _kSampleAsset = 'assets/samples/aerial_sample.jpg';

class _PhotoDetectViewState extends State<PhotoDetectView> {
  final ImagePicker _picker = ImagePicker();

  ui.Image? _image;
  Size _imageSize = Size.zero;
  List<Detection> _detections = const <Detection>[];
  FrameTimings? _timings;

  bool _working = false;
  String? _error;

  Future<void> _pick() async {
    if (_working) return;
    setState(() {
      _working = true;
      _error = null;
    });
    try {
      final XFile? file = await _picker.pickImage(source: ImageSource.gallery);
      if (file == null) {
        if (mounted) setState(() => _working = false);
        return;
      }
      await _runOnBytes(await file.readAsBytes());
    } catch (e) {
      if (mounted) {
        setState(() {
          _working = false;
          _error = '$e';
        });
      }
    }
  }

  /// Load the bundled sample image and run it through the exact same still path
  /// as [_pick] — no photo-library access required.
  Future<void> _trySample() async {
    if (_working) return;
    setState(() {
      _working = true;
      _error = null;
    });
    try {
      final ByteData data = await rootBundle.load(_kSampleAsset);
      await _runOnBytes(data.buffer.asUint8List());
    } catch (e) {
      if (mounted) {
        setState(() {
          _working = false;
          _error = '$e';
        });
      }
    }
  }

  /// Shared still-image pipeline for both the library pick and the bundled
  /// sample: decode (EXIF baked, off the UI isolate), build the display image,
  /// and run [MelangeService.inferStill]. Single implementation so the two entry
  /// points can never drift apart.
  Future<void> _runOnBytes(Uint8List fileBytes) async {
    final DecodedImage? decoded = await decodeStillImage(fileBytes);
    if (decoded == null) {
      if (mounted) {
        setState(() {
          _working = false;
          _error = 'Could not decode that image.';
        });
      }
      return;
    }

    // Build the display image and run inference.
    final ui.Image image = await _toUiImage(decoded);
    final result = await widget.service.inferStill(
      decoded.rgb,
      decoded.width,
      decoded.height,
    );
    if (!mounted) return;

    // Replace the previous display image, if any.
    _image?.dispose();
    setState(() {
      _image = image;
      _imageSize = Size(decoded.width.toDouble(), decoded.height.toDouble());
      _detections = result?.detections ?? const <Detection>[];
      _timings = result?.timings;
      _working = false;
      _error = result == null ? 'Model not ready — try again.' : null;
    });
  }

  Future<ui.Image> _toUiImage(DecodedImage d) {
    final Completer<ui.Image> completer = Completer<ui.Image>();
    ui.decodeImageFromPixels(
      d.rgba,
      d.width,
      d.height,
      ui.PixelFormat.rgba8888,
      completer.complete,
    );
    return completer.future;
  }

  @override
  void dispose() {
    _image?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ui.Image? image = _image;

    return Stack(
      fit: StackFit.expand,
      children: <Widget>[
        if (image != null)
          PhotoOverlay(image: image, detections: _detections)
        else
          _EmptyState(error: _error),

        if (image != null)
          Hud(
            detections: _detections,
            timings: _timings,
            bufferSize: _imageSize,
            sensorOrientation: 0,
          ),

        // Pick / re-pick + bundled-sample controls.
        Positioned(
          left: 0,
          right: 0,
          bottom: 24,
          child: Center(
            child: Wrap(
              spacing: 12,
              runSpacing: 12,
              alignment: WrapAlignment.center,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: <Widget>[
                FilledButton.icon(
                  onPressed: _working ? null : _pick,
                  icon: _working
                      ? const SizedBox(
                          width: 16,
                          height: 16,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Icon(Icons.photo_library),
                  label: Text(image == null ? 'Pick image' : 'Pick another'),
                ),
                OutlinedButton.icon(
                  onPressed: _working ? null : _trySample,
                  icon: const Icon(Icons.auto_awesome),
                  label: const Text('Try sample'),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({this.error});

  final String? error;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            Icon(
              error == null ? Icons.image_search : Icons.error_outline,
              size: 56,
              color: error == null ? Colors.white38 : Colors.redAccent,
            ),
            const SizedBox(height: 12),
            Text(
              error ??
                  'Pick an aerial or drone photo to run detection.\n'
                      'SkyScout is trained on top-down imagery.',
              textAlign: TextAlign.center,
              style: TextStyle(
                color: error == null ? Colors.white54 : Colors.redAccent,
                fontSize: 13,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/services.dart';

/// One bundled echo clip: golden input tensors (identical to the FP32 export
/// goldens, so device-vs-FP32 IoU isolates quantization drift), their golden LV
/// masks, and the display frames.
class ClipData {
  ClipData({
    required this.id,
    required this.label,
    required this.view,
    required this.numFrames,
    required this.inputC,
    required this.inputH,
    required this.inputW,
    required this.displaySize,
    required this.inputs,
    required this.goldenMasks,
    required this.framePaths,
    required this.attribution,
  });

  final String id, label, view;
  final int numFrames;
  final int inputC, inputH, inputW;
  final int displaySize;
  final List<Float32List> inputs; // flat NCHW per frame
  final List<Uint8List> goldenMasks; // per frame, 0/1
  final List<String> framePaths; // display frame asset keys
  final String attribution;

  int get maskLen => inputH * inputW;
  String get thumbPath => framePaths.isEmpty ? '' : framePaths.first;
}

/// The set of demo clips selectable in the app.
class ClipLibrary {
  ClipLibrary(this.clips);
  final List<ClipData> clips;

  int get inputC => clips.first.inputC;
  int get inputH => clips.first.inputH;
  int get inputW => clips.first.inputW;

  /// [clip][frame] input tensors, for the inference worker.
  List<List<Float32List>> get inputsByClip => [for (final c in clips) c.inputs];

  static Future<ClipLibrary> load() async {
    const dir = 'assets/clips';
    final manifest =
        jsonDecode(await rootBundle.loadString('$dir/manifest.json')) as Map;
    final inShape = (manifest['input']['shape'] as List)
        .cast<int>(); // [1,3,112,112]
    final c = inShape[1], h = inShape[2], w = inShape[3];
    final display = manifest['display']['size'] as int;
    final inStride = c * h * w;
    final maskStride = h * w;

    final clips = <ClipData>[];
    for (final raw in (manifest['clips'] as List)) {
      final m = raw as Map;
      final id = m['id'] as String;
      final n = m['num_frames'] as int;
      final base = '$dir/$id';

      final allInputs = await _loadFloat32('$base/inputs.bin');
      final inputs = List<Float32List>.generate(
        n,
        (i) => Float32List.sublistView(
          allInputs,
          i * inStride,
          (i + 1) * inStride,
        ),
      );
      final allMasks = await _loadUint8('$base/masks.bin');
      final masks = List<Uint8List>.generate(
        n,
        (i) => Uint8List.sublistView(
          allMasks,
          i * maskStride,
          (i + 1) * maskStride,
        ),
      );
      final paths = List<String>.generate(
        n,
        (i) => '$base/frame_${i.toString().padLeft(2, '0')}.png',
      );

      clips.add(
        ClipData(
          id: id,
          label: m['label'] as String,
          view: m['view'] as String,
          numFrames: n,
          inputC: c,
          inputH: h,
          inputW: w,
          displaySize: display,
          inputs: inputs,
          goldenMasks: masks,
          framePaths: paths,
          attribution: (m['attribution'] ?? '') as String,
        ),
      );
    }
    return ClipLibrary(clips);
  }

  static Future<Float32List> _loadFloat32(String key) async {
    final bd = await rootBundle.load(key);
    // Copy into a fresh (aligned) buffer before the Float32List view.
    final u8 = Uint8List.fromList(
      bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes),
    );
    return u8.buffer.asFloat32List();
  }

  static Future<Uint8List> _loadUint8(String key) async {
    final bd = await rootBundle.load(key);
    return Uint8List.fromList(
      bd.buffer.asUint8List(bd.offsetInBytes, bd.lengthInBytes),
    );
  }
}

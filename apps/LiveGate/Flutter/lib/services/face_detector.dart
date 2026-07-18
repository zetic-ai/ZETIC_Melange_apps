import 'dart:math' show Point;
import 'dart:typed_data';
import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../models/geometry.dart';

/// One detected face: its bounding box and 5 alignment landmarks, both in the
/// upright image space (the space produced by rotating the frame by
/// [rotationDegrees], which must match [frameToUprightBgr]).
class DetectedFace {
  const DetectedFace({required this.box, required this.landmarks});
  final FaceBox box;
  final Landmarks5 landmarks;
}

/// Thin wrapper over ML Kit on-device face detection. Fully offline (the model
/// is bundled), so it needs no network and keeps the "on-device · no cloud"
/// badge honest — the reason it was chosen over reusing the repo's native-only
/// MediaPipe Melange demos (which would add 2 more model loads). See HANDOFF.md.
class FaceDetectorService {
  FaceDetectorService()
      : _detector = FaceDetector(
          options: FaceDetectorOptions(
            performanceMode: FaceDetectorMode.fast,
            enableLandmarks: true,
            enableContours: false,
            enableClassification: false,
          ),
        );

  final FaceDetector _detector;

  /// Detects the largest face in [image]. [rotationDegrees] (0/90/180/270) must
  /// match the rotation handed to [frameToUprightBgr] so the returned
  /// coordinates share the model's upright space. Returns null if no usable
  /// face (or a face missing any of the 5 landmarks) is found.
  Future<DetectedFace?> detect(
    CameraImage image,
    int rotationDegrees,
  ) async {
    final input = _toInputImage(image, rotationDegrees);
    if (input == null) return null;

    final faces = await _detector.processImage(input);
    if (faces.isEmpty) return null;

    // Largest face by box area.
    Face best = faces.first;
    for (final f in faces) {
      if (f.boundingBox.width * f.boundingBox.height >
          best.boundingBox.width * best.boundingBox.height) {
        best = f;
      }
    }

    final le = best.landmarks[FaceLandmarkType.leftEye]?.position;
    final re = best.landmarks[FaceLandmarkType.rightEye]?.position;
    final nose = best.landmarks[FaceLandmarkType.noseBase]?.position;
    final ml = best.landmarks[FaceLandmarkType.leftMouth]?.position;
    final mr = best.landmarks[FaceLandmarkType.rightMouth]?.position;
    if (le == null || re == null || nose == null || ml == null || mr == null) {
      return null;
    }

    // Order eyes and mouth corners by IMAGE x (not ML Kit's subject-relative
    // left/right label). The ArcFace template's slot 0 is the image-left eye,
    // slot 1 the image-right eye, etc. A similarity transform cannot reflect, so
    // pairing by image position — rather than by semantic label — makes the warp
    // robust to any mirroring convention.
    final eyes = [_pt(le), _pt(re)]..sort((a, b) => a.x.compareTo(b.x));
    final mouths = [_pt(ml), _pt(mr)]..sort((a, b) => a.x.compareTo(b.x));

    final r = best.boundingBox;
    return DetectedFace(
      box: FaceBox(
        x: r.left,
        y: r.top,
        w: r.width,
        h: r.height,
      ),
      landmarks: Landmarks5(
        leftEye: eyes[0],
        rightEye: eyes[1],
        nose: _pt(nose),
        mouthLeft: mouths[0],
        mouthRight: mouths[1],
      ),
    );
  }

  Point2 _pt(Point<int> p) => Point2(p.x.toDouble(), p.y.toDouble());

  InputImage? _toInputImage(CameraImage image, int rotationDegrees) {
    final rotation =
        InputImageRotationValue.fromRawValue(rotationDegrees) ??
            InputImageRotation.rotation0deg;

    if (image.format.group == ImageFormatGroup.bgra8888) {
      final plane = image.planes.first;
      return InputImage.fromBytes(
        // Copy out of the recycled camera buffer; ML Kit reads it async.
        bytes: Uint8List.fromList(plane.bytes),
        metadata: InputImageMetadata(
          size: Size(image.width.toDouble(), image.height.toDouble()),
          rotation: rotation,
          format: InputImageFormat.bgra8888,
          bytesPerRow: plane.bytesPerRow,
        ),
      );
    }

    // Android YUV420 -> NV21 (the format ML Kit accepts as a single buffer).
    final nv21 = _yuv420ToNv21(image);
    return InputImage.fromBytes(
      bytes: nv21,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: InputImageFormat.nv21,
        bytesPerRow: image.width,
      ),
    );
  }

  /// Packs a 3-plane YUV_420_888 [CameraImage] into a single NV21 buffer
  /// (full-res Y followed by interleaved V,U at half resolution), honoring row
  /// and pixel strides.
  static Uint8List _yuv420ToNv21(CameraImage image) {
    final int width = image.width;
    final int height = image.height;
    final out = Uint8List(width * height + (width * height ~/ 2));

    final Plane yPlane = image.planes[0];
    final Plane uPlane = image.planes[1];
    final Plane vPlane = image.planes[2];

    // Y.
    int o = 0;
    final yBytes = yPlane.bytes;
    final int yRowStride = yPlane.bytesPerRow;
    for (var row = 0; row < height; row++) {
      final int base = row * yRowStride;
      for (var col = 0; col < width; col++) {
        out[o++] = yBytes[base + col];
      }
    }

    // Interleaved VU (NV21) at half res.
    final uBytes = uPlane.bytes;
    final vBytes = vPlane.bytes;
    final int uvRowStride = uPlane.bytesPerRow;
    final int uvPixelStride = uPlane.bytesPerPixel ?? 1;
    for (var row = 0; row < height ~/ 2; row++) {
      final int base = row * uvRowStride;
      for (var col = 0; col < width ~/ 2; col++) {
        final int uvIndex = base + col * uvPixelStride;
        out[o++] = vBytes[uvIndex];
        out[o++] = uBytes[uvIndex];
      }
    }
    return out;
  }

  void dispose() => _detector.close();
}

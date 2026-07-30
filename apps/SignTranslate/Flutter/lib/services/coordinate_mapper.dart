import 'dart:math' as math;
import 'dart:ui' show Offset, Size;

/// Maps a point in upright-frame space (`frameW`×`frameH` — the space every
/// tracked quad lives in) to screen space, assuming the camera preview is
/// rendered BoxFit.cover into [screen] (center-cropped, aspect preserved).
///
/// Pure and unit-tested: the PyroGuard orientation bug was an overlay
/// applying a spurious extra rotation — this mapper does scale+center only;
/// orientation is fully handled upstream (the frame is already upright).
Offset mapFrameToScreen(
  Offset framePoint,
  double frameW,
  double frameH,
  Size screen,
) {
  final scale = math.max(screen.width / frameW, screen.height / frameH);
  final dx = (screen.width - frameW * scale) / 2;
  final dy = (screen.height - frameH * scale) / 2;
  return Offset(framePoint.dx * scale + dx, framePoint.dy * scale + dy);
}

/// Exact inverse of [mapFrameToScreen] (round-trip tested).
Offset mapScreenToFrame(
  Offset screenPoint,
  double frameW,
  double frameH,
  Size screen,
) {
  final scale = math.max(screen.width / frameW, screen.height / frameH);
  final dx = (screen.width - frameW * scale) / 2;
  final dy = (screen.height - frameH * scale) / 2;
  return Offset((screenPoint.dx - dx) / scale, (screenPoint.dy - dy) / scale);
}

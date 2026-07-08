import 'dart:typed_data';

/// Intersection-over-union of two binary masks of equal length.
/// [pred] is derived from device logits (> 0); [golden] is the FP32 reference
/// (0/1). Empty-vs-empty returns 1.0 (perfect agreement on "no LV" is correct).
double maskIoU(Uint8List pred, Uint8List golden) {
  assert(pred.length == golden.length);
  var inter = 0, union = 0;
  for (var i = 0; i < pred.length; i++) {
    final a = pred[i] != 0;
    final b = golden[i] != 0;
    if (a && b) inter++;
    if (a || b) union++;
  }
  if (union == 0) return 1.0;
  return inter / union;
}

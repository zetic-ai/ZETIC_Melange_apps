import 'dart:typed_data';

/// Model context window length (must exactly match the exported ONNX).
const int kContextLength = 512;

/// Fixed-capacity ring buffer over the most recent [kContextLength] sensor
/// samples. Pre-allocated once; no per-tick allocation.
///
/// CONTRACT (SPEC "full-window"): the model may only run once [isFull] is
/// true, and the tensor is the raw values — NO normalization, scaling, or
/// padding of any kind (instance norm + de-norm live inside the ONNX graph).
class SampleWindow {
  SampleWindow([this.capacity = kContextLength])
      : _buf = Float32List(capacity);

  final int capacity;
  final Float32List _buf;
  int _next = 0; // write position
  int _count = 0; // total samples ever written (saturates display at capacity)

  bool get isFull => _count >= capacity;

  /// Total samples pushed since creation (== global index of next sample).
  int get totalPushed => _count;

  void push(double value) {
    _buf[_next] = value;
    _next = (_next + 1) % capacity;
    _count++;
  }

  /// Copies the window, oldest -> newest, into [dst] (length [capacity]).
  /// Throws [StateError] if the window is not yet full: inference must never
  /// run on a partial window (the export has no padding support).
  void snapshotInto(Float32List dst) {
    if (!isFull) {
      throw StateError(
          'SampleWindow.snapshotInto called with only $_count/$capacity samples');
    }
    if (dst.length != capacity) {
      throw ArgumentError('dst.length ${dst.length} != capacity $capacity');
    }
    // Oldest sample sits at _next (the slot about to be overwritten).
    final tail = capacity - _next;
    dst.setRange(0, tail, _buf, _next);
    dst.setRange(tail, capacity, _buf, 0);
  }
}

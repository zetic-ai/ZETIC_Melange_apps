import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voxscribe/services/postprocessor.dart' show kMaxLen;
import 'package:zetic_mlange/zetic_mlange.dart';

/// A12 — token dtype/byte-length. ids and enc_mask are int32 (4-byte), length
/// 448 => 1792 bytes each, built via Tensor.int32List with DataType.int32.
/// (Constructing a Tensor is pure Dart; the native lib loads lazily on
/// model create/run, so this runs on the host with no device.)
void main() {
  test('decoder ids/mask tensors are int32, 448 elems, 1792 bytes', () {
    expect(kMaxLen, 448);
    final Int32List ids = Int32List(kMaxLen)..fillRange(0, kMaxLen, 50256);
    ids[0] = 50258; // SOT
    final Int32List mask = Int32List(kMaxLen);
    mask[0] = 1;

    final Tensor idsT = Tensor.int32List(ids, shape: <int>[1, kMaxLen]);
    final Tensor maskT = Tensor.int32List(mask, shape: <int>[1, kMaxLen]);

    expect(idsT.dataType, DataType.int32);
    expect(maskT.dataType, DataType.int32);
    expect(idsT.count(), 448);
    expect(maskT.count(), 448);
    expect(idsT.byteLength, 448 * 4); // 1792
    expect(maskT.byteLength, 1792);
    expect(DataType.int32.bytesPerElement, 4);
  });
}

import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_secure_storage/flutter_secure_storage.dart';

/// Stores the enrolled reference identity as a single L2-normalized 128-d
/// embedding in the platform secure store (iOS Keychain / Android Keystore).
///
/// Privacy invariant (SPEC_STUB.md): **only the vector is persisted, never a
/// face image.** The vector is not reversible to a photo.
class Enrollment {
  Enrollment({FlutterSecureStorage? storage})
      : _storage = storage ?? const FlutterSecureStorage();

  static const String _key = 'livegate.enrolled.embedding.v1';

  final FlutterSecureStorage _storage;

  /// In-memory cache of the enrolled vector so the hot path does not hit secure
  /// storage every frame.
  Float32List? _cached;
  bool _loaded = false;

  bool get isEnrolled => _cached != null;
  Float32List? get vector => _cached;

  /// Loads the enrolled vector from secure storage into the cache. Call once at
  /// startup.
  Future<Float32List?> load() async {
    final raw = await _storage.read(key: _key);
    _cached = raw == null ? null : _decode(raw);
    _loaded = true;
    return _cached;
  }

  /// Persists [normalized] (must already be L2-normalized) as the reference.
  Future<void> enroll(Float32List normalized) async {
    _cached = Float32List.fromList(normalized);
    await _storage.write(key: _key, value: _encode(normalized));
  }

  /// Removes the enrolled reference.
  Future<void> clear() async {
    _cached = null;
    await _storage.delete(key: _key);
  }

  bool get isLoaded => _loaded;

  static String _encode(Float32List v) =>
      base64Encode(v.buffer.asUint8List(v.offsetInBytes, v.lengthInBytes));

  static Float32List _decode(String s) {
    final bytes = base64Decode(s);
    // Copy into a fresh, aligned buffer (the base64 buffer may not be aligned).
    final copy = Uint8List.fromList(bytes);
    return copy.buffer.asFloat32List();
  }
}

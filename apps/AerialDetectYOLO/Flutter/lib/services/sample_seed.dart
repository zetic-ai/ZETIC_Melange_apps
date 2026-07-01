import 'package:flutter/services.dart' show Uint8List, rootBundle;
import 'package:gal/gal.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Bundled demo aerial image (a drone car-park shot) shipped with the app so the
/// detector can be exercised without a photo library — see pubspec `assets`.
const String _kSampleAsset = 'assets/samples/aerial_sample.jpg';

/// Persisted "the demo image has already been seeded into Photos" flag.
const String _kSampleSavedKey = 'skyscout_sample_saved';

/// Silently place the bundled demo aerial image into the device photo library
/// exactly once, ever, so the user can upload it themselves — no button, no
/// snackbar, no UI.
///
/// A [SharedPreferences] flag (`skyscout_sample_saved`) guards against ever
/// duplicating the image: it is only set after a successful save, so a denied
/// permission simply leaves it unset (a later launch may retry) rather than
/// baking in a failure. Everything is wrapped so a denied permission or plugin
/// error can never crash or block app start. On iOS the standard one-time
/// add-to-Photos prompt (backed by `NSPhotoLibraryAddUsageDescription`) on the
/// first launch is expected.
Future<void> saveSampleToPhotosOnce() async {
  try {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    if (prefs.getBool(_kSampleSavedKey) ?? false) return;

    final Uint8List bytes =
        (await rootBundle.load(_kSampleAsset)).buffer.asUint8List();
    await Gal.putImageBytes(bytes, name: 'skyscout_demo');

    await prefs.setBool(_kSampleSavedKey, true);
  } catch (_) {
    // Denied permission or any other error: swallow it. The unset flag lets a
    // future launch retry, and nothing is ever surfaced to the user.
  }
}

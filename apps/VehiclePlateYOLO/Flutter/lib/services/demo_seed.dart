import 'package:flutter/services.dart' show rootBundle;
import 'package:gal/gal.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Asset path of the bundled demo still.
const String _kSampleAsset = 'assets/samples/plate_sample.jpg';

/// Persisted flag key: set once the demo image has been copied to the gallery.
const String _kSavedFlag = 'platehawk_sample_saved';

/// Silently copies the bundled demo photo into the device's photo library the
/// FIRST time the app is ever launched, so the user has something to upload
/// without any in-app "demo" button.
///
/// Guaranteed to run its side effect at most once per install: a
/// [SharedPreferences] flag ([_kSavedFlag]) is checked first and only set after
/// a successful save, so later launches never duplicate the image in the camera
/// roll. iOS shows the standard one-time add-to-Photos permission prompt
/// (NSPhotoLibraryAddUsageDescription); Android already has the image seeded via
/// adb but re-saving here is harmless thanks to the once-flag.
///
/// Everything is wrapped in try/catch: a denied Photos permission or any other
/// error is swallowed so this never crashes or blocks app start. Fire-and-forget.
Future<void> ensureDemoImageSaved() async {
  try {
    final prefs = await SharedPreferences.getInstance();
    if (prefs.getBool(_kSavedFlag) == true) return;

    final data = await rootBundle.load(_kSampleAsset);
    await Gal.putImageBytes(
      data.buffer.asUint8List(),
      name: 'platehawk_demo',
    );

    await prefs.setBool(_kSavedFlag, true);
  } catch (_) {
    // Denied permission or any failure: ignore. We simply retry on next launch
    // (the flag is only set on success), and the app continues normally.
  }
}

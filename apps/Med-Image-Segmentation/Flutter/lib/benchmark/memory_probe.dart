import 'package:flutter/services.dart';

/// Reads the process memory footprint via a platform channel (the plugin exposes
/// no memory API). iOS returns task_info phys_footprint; Android returns
/// Debug.MemoryInfo().totalPss. Both in MB. Returns 0 if unavailable.
class MemoryProbe {
  static const _channel = MethodChannel('medseg/memory');

  static Future<double> footprintMB() async {
    try {
      final mb = await _channel.invokeMethod<double>('footprintMB');
      return mb ?? 0;
    } on PlatformException {
      return 0;
    } on MissingPluginException {
      return 0;
    }
  }
}

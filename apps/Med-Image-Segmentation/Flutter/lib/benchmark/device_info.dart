import 'dart:io';

import 'package:device_info_plus/device_info_plus.dart';

/// Identity for the benchmark row: the device the demo is running on.
class DeviceIdentity {
  DeviceIdentity({required this.model, required this.chip, required this.os});
  final String model; // e.g. "iPhone14,5" / "SM-S921N"
  final String
  chip; // best-effort; iOS gives model id, Android gives SoC/hardware
  final String os; // e.g. "iOS 18.2" / "Android 16"

  String get short => '$model · $os';

  static Future<DeviceIdentity> load() async {
    final info = DeviceInfoPlugin();
    if (Platform.isIOS) {
      final ios = await info.iosInfo;
      return DeviceIdentity(
        model: ios.utsname.machine, // hardware id, e.g. iPhone16,1
        chip: ios
            .utsname
            .machine, // Apple doesn't expose SoC name; model id maps to it
        os: '${ios.systemName} ${ios.systemVersion}',
      );
    }
    if (Platform.isAndroid) {
      final a = await info.androidInfo;
      final chip = [a.hardware, a.board].where((s) => s.isNotEmpty).join('/');
      return DeviceIdentity(
        model: '${a.manufacturer} ${a.model}'.trim(),
        chip: chip.isEmpty ? a.device : chip,
        os: 'Android ${a.version.release}',
      );
    }
    return DeviceIdentity(
      model: 'unknown',
      chip: 'unknown',
      os: Platform.operatingSystem,
    );
  }
}

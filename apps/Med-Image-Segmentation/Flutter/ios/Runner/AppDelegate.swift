import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)

    // medseg/memory — the plugin exposes no memory API, so report the process
    // footprint (the metric jetsam uses) for the benchmark's peak-memory column.
    if let registrar = engineBridge.pluginRegistry.registrar(forPlugin: "MedSegMemory") {
      let channel = FlutterMethodChannel(
        name: "medseg/memory", binaryMessenger: registrar.messenger())
      channel.setMethodCallHandler { call, reply in
        if call.method == "footprintMB" {
          reply(AppDelegate.footprintMB())
        } else {
          reply(FlutterMethodNotImplemented)
        }
      }
    }
  }

  private static func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) { ptr in
      ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return Double(info.phys_footprint) / 1_048_576.0
  }
}

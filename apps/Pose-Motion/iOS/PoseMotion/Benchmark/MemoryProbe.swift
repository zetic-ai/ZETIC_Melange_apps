import Foundation

/// Reports the process memory footprint (the metric the OS jetsam uses), for
/// confirming on-device that memory plateaus rather than climbing. DEBUG-only logging.
enum MemoryProbe {
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576.0
    }

    static func log(_ tag: String) {
        #if DEBUG
        print(String(format: "[mem] %-16@ %.0f MB", tag as NSString, footprintMB()))
        #endif
    }
}

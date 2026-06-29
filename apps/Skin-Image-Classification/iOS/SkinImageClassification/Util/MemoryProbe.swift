import Foundation
import OSLog

/// Lightweight resident-memory probe. The MedGemma-4b weights (~2.5 GB) plus the
/// classifier sit close to the jetsam limit on 4–6 GB devices, so we log peak RSS
/// at each pipeline stage to catch a kill before a live demo does.
enum MemoryProbe {
    private static let log = Logger(subsystem: "ai.zetic.demo.SkinImageClassification", category: "memory")

    /// Current resident footprint in megabytes, or nil if unavailable.
    static func residentMB() -> Double? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return Double(info.phys_footprint) / 1_048_576
    }

    /// Log footprint with a stage label, e.g. `MemoryProbe.log("after LLM load")`.
    static func log(_ stage: String) {
        if let mb = residentMB() {
            log.info("RSS \(String(format: "%.0f", mb), privacy: .public) MB — \(stage, privacy: .public)")
        }
    }
}

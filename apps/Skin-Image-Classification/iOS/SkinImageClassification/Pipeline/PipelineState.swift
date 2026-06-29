import Foundation

/// Generic load lifecycle for an on-device model, shared by the classifier and the
/// LLM download UI. `downloading` carries real file progress; `preparing` covers
/// compile/init, for which the SDK reports no progress (UI shows an indeterminate
/// state rather than a frozen 0%).
enum LoadPhase: Equatable {
    case idle
    case downloading(Double)   // 0…1
    case preparing
    case ready
    case failed(String)

    var isReady: Bool { self == .ready }

    /// 0…1 for a determinate bar, or nil when indeterminate.
    var progress: Double? {
        if case .downloading(let v) = self { return v }
        return nil
    }

    var errorMessage: String? {
        if case .failed(let m) = self { return m }
        return nil
    }
}

/// The per-photo analysis lifecycle, after models are ready.
enum AnalysisState: Equatable {
    case none          // awaiting a photo
    case classifying   // running the vision model
    case explaining    // classification shown; MedGemma streaming
    case done
    case failed(String)

    /// True once we have a classification to display (card visible).
    var hasResult: Bool {
        switch self {
        case .explaining, .done: return true
        default: return false
        }
    }
}

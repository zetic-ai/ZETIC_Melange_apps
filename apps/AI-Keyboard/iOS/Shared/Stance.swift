import Foundation

/// Reply stance offered as chips under the Reply action.
enum Stance: String, Codable, CaseIterable, Identifiable, Hashable {
    case agreeable
    case disagreeable

    var id: String { rawValue }

    /// Chip label.
    var label: String {
        switch self {
        case .agreeable:    return "Agreeable"
        case .disagreeable: return "Disagreeable"
        }
    }

    /// Phrase injected into the reply prompt.
    var descriptor: String {
        switch self {
        case .agreeable:    return "agreeable and supportive"
        case .disagreeable: return "politely disagreeing"
        }
    }
}

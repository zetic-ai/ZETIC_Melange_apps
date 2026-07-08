import Foundation

/// Rewrite tones offered as chips under the Rewrite action.
enum Tone: String, Codable, CaseIterable, Identifiable, Hashable {
    case professional
    case casual
    case friendly
    case romantic

    var id: String { rawValue }

    /// Chip label.
    var label: String {
        switch self {
        case .professional: return "Professional"
        case .casual:       return "Casual"
        case .friendly:     return "Friendly"
        case .romantic:     return "Romantic"
        }
    }

    /// Phrase injected into the rewrite prompt.
    var descriptor: String {
        switch self {
        case .professional: return "polished and professional"
        case .casual:       return "relaxed and casual"
        case .friendly:     return "warm and friendly"
        case .romantic:     return "affectionate and romantic"
        }
    }
}

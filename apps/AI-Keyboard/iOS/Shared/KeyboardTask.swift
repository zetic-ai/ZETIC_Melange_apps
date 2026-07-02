import Foundation

/// The four AI actions CherryPad offers, mirroring MangoPad. `Codable` so the
/// keyboard extension can encode a chosen task into the handoff payload.
enum KeyboardTask: String, Codable, CaseIterable, Identifiable, Hashable {
    case rewrite
    case reply
    case translate
    case grammar

    var id: String { rawValue }

    /// Action-bar label.
    var title: String {
        switch self {
        case .rewrite:   return "Rewrite"
        case .reply:     return "Reply"
        case .translate: return "Translate"
        case .grammar:   return "Grammar"
        }
    }

    /// SF Symbol shown on the action chip.
    var symbol: String {
        switch self {
        case .rewrite:   return "arrow.triangle.2.circlepath"
        case .reply:     return "bubble.left.and.bubble.right.fill"
        case .translate: return "globe"
        case .grammar:   return "checkmark.seal.fill"
        }
    }

    /// One-line tagline echoing MangoPad's marketing copy.
    var tagline: String {
        switch self {
        case .rewrite:   return "Better words, same you."
        case .reply:     return "Let it reply, reduce mental load."
        case .translate: return "Cross language barriers."
        case .grammar:   return "Fix grammar, keep your tone."
        }
    }
}

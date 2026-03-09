import Foundation
import SwiftUI

enum Role: String, Codable { case user, assistant }

struct Message: Identifiable, Codable, Hashable {
    let id: UUID
    let role: Role
    var content: String
    let timestamp: Date
    let source: String
    let target: String
}

struct Session: Identifiable, Codable, Hashable {
    let id: UUID
    var title: String
    var updatedAt: Date
    var messages: [Message]
}

struct FavoritePhrase: Identifiable, Codable, Hashable {
    let id: UUID
    let text: String
    let source: Language
    let target: Language
    let createdAt: Date
}

struct DiagnosticsSnapshot: Codable, Hashable {
    var lastRun: Date?
    var lastDurationMs: Int
    var lastTokenCount: Int
    var lastRawLog: String
    var lastStopReason: String

    static let empty = DiagnosticsSnapshot(lastRun: nil, lastDurationMs: 0, lastTokenCount: 0, lastRawLog: "", lastStopReason: "idle")
}

struct GenerationRecord {
    let text: String
    let tokenCount: Int
    let durationMs: Int
    let stopped: Bool
    let error: String?
}

struct SettingsData: Codable, Hashable {
    enum Theme: String, Codable, CaseIterable {
        case system, light, dark
        var colorScheme: ColorScheme? {
            switch self {
            case .system: return nil
            case .light: return .light
            case .dark: return .dark
            }
        }
        var label: String { rawValue.capitalized }
    }

    var theme: Theme
    var defaultSource: String
    var defaultTarget: String
    var systemPrompt: String
}

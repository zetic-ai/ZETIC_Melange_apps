import Foundation

enum Role: String, Codable { case user, assistant }

struct Message: Codable, Identifiable, Hashable {
    let id: UUID
    let role: Role
    var content: String
    let timestamp: Date
}

struct Session: Codable, Identifiable, Hashable {
    let id: UUID
    var title: String
    var updatedAt: Date
    var messages: [Message]
}

struct DiagnosticsSnapshot: Codable {
    var lastRun: Date?
    var lastDurationMs: Int
    var lastTokenCount: Int
    var lastRawLog: String
    var lastStopReason: String
}

struct GenerationRecord {
    let text: String
    let tokenCount: Int
    let durationMs: Int
    let stopped: Bool
    let error: String?
}

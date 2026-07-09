import Foundation

/// Request written by the keyboard extension and read by the container app.
/// `tone` / `stance` / `targetLanguage` are optional — the keyboard hands off the
/// captured text and task; the app pre-selects sensible defaults and lets the user
/// adjust tone/stance/language before (re)running.
struct HandoffRequest: Codable, Equatable {
    let id: UUID
    let task: KeyboardTask
    let text: String
    var tone: Tone?
    var stance: Stance?
    var targetLanguage: String?   // Language.englishName, or nil → app default
    let createdAt: Date

    init(id: UUID = UUID(),
         task: KeyboardTask,
         text: String,
         tone: Tone? = nil,
         stance: Stance? = nil,
         targetLanguage: String? = nil,
         createdAt: Date = Date()) {
        self.id = id
        self.task = task
        self.text = text
        self.tone = tone
        self.stance = stance
        self.targetLanguage = targetLanguage
        self.createdAt = createdAt
    }
}

/// Result written back by the container app for the keyboard to insert.
struct HandoffResult: Codable, Equatable {
    let requestID: UUID
    let text: String
    let createdAt: Date

    init(requestID: UUID, text: String, createdAt: Date = Date()) {
        self.requestID = requestID
        self.text = text
        self.createdAt = createdAt
    }
}

/// Read/write helpers over the shared App Group defaults. Both sides go through
/// this so the keys never drift.
enum HandoffStore {
    private static let requestKey = "cherrypad.request"
    private static let resultKey = "cherrypad.result"

    // MARK: Request (keyboard → app)

    static func writeRequest(_ request: HandoffRequest) {
        guard let data = try? JSONEncoder().encode(request) else { return }
        AppGroup.defaults.set(data, forKey: requestKey)
    }

    static func readRequest() -> HandoffRequest? {
        guard let data = AppGroup.defaults.data(forKey: requestKey) else { return nil }
        return try? JSONDecoder().decode(HandoffRequest.self, from: data)
    }

    static func clearRequest() {
        AppGroup.defaults.removeObject(forKey: requestKey)
    }

    // MARK: Result (app → keyboard)

    static func writeResult(_ result: HandoffResult) {
        guard let data = try? JSONEncoder().encode(result) else { return }
        AppGroup.defaults.set(data, forKey: resultKey)
    }

    static func readResult() -> HandoffResult? {
        guard let data = AppGroup.defaults.data(forKey: resultKey) else { return nil }
        return try? JSONDecoder().decode(HandoffResult.self, from: data)
    }

    static func clearResult() {
        AppGroup.defaults.removeObject(forKey: resultKey)
    }
}

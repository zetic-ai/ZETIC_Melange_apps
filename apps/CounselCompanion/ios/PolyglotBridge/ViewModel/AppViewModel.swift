import Foundation
import SwiftUI

@MainActor
final class AppViewModel: ObservableObject {
    @Published var sessions: [Session] = []
    @Published var favorites: [FavoritePhrase] = []
    @Published var settings: SettingsData
    @Published var diagnostics: DiagnosticsSnapshot = .empty
    @Published var currentSessionID: UUID?
    @Published var modelReady: Bool
    @Published var preparationProgress: Double = 0

    let languages = Language.catalog

    private let historyStore = HistoryStore()
    private let favoritesStore = FavoritesStore()
    private let settingsStore = SettingsStore()
    private let promptBuilder = PromptBuilder()
    private let engine = TranslationEngine()
    private let prepManager = ModelPreparationManager()

    init() {
        self.sessions = historyStore.load()
        self.favorites = favoritesStore.load()
        self.settings = settingsStore.load()
        self.modelReady = settingsStore.modelReady
        if sessions.isEmpty { createNewSession() }
        else { currentSessionID = sessions.first?.id }
    }

    func createNewSession() {
        guard !isGenerating else { return }
        let session = Session(id: UUID(), title: "New Session", updatedAt: Date(), messages: [])
        sessions.insert(session, at: 0)
        currentSessionID = session.id
        persistSessions()
    }

    func deleteSession(_ id: UUID) {
        sessions.removeAll { $0.id == id }
        if currentSessionID == id { currentSessionID = sessions.first?.id }
        persistSessions()
    }

    func resumeSession(_ id: UUID) {
        currentSessionID = id
    }

    func renameSession(_ id: UUID, title: String) {
        guard let idx = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[idx].title = title
        sessions[idx].updatedAt = Date()
        persistSessions()
    }

    func saveFavorite(from message: Message, source: Language, target: Language) {
        let fav = FavoritePhrase(id: UUID(), text: message.content, source: source, target: target, createdAt: Date())
        favorites.insert(fav, at: 0)
        favoritesStore.save(favorites)
    }

    func removeFavorite(_ id: UUID) {
        favorites.removeAll { $0.id == id }
        favoritesStore.save(favorites)
    }

    func clearFavorites() {
        favorites = []
        favoritesStore.clear()
    }

    func updateSettings(_ block: (inout SettingsData) -> Void) {
        block(&settings)
        settingsStore.save(settings)
    }

    func clearHistory() {
        sessions = []
        historyStore.clear()
        createNewSession()
    }

    func markModelReady(_ ready: Bool) {
        modelReady = ready
        settingsStore.modelReady = ready
    }

    // MARK: - Translation
    @Published private(set) var isGenerating = false
    @Published private(set) var loadingMessage: String = "Ready"
    @Published private(set) var errorMessage: String?

    func send(text: String, source: Language, target: Language) {
        guard let sessionID = currentSessionID else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isGenerating else { return }
        let userMessage = Message(id: UUID(), role: .user, content: trimmed, timestamp: Date(), source: source.code, target: target.code)
        appendMessage(userMessage, to: sessionID)
        let assistant = Message(id: UUID(), role: .assistant, content: "", timestamp: Date(), source: source.code, target: target.code)
        appendMessage(assistant, to: sessionID)

        let history = messages(for: sessionID)
        let prompt = promptBuilder.buildPrompt(systemPrompt: settings.systemPrompt, history: Array(history.dropLast()), userInput: trimmed, source: source, target: target)

        isGenerating = true
        loadingMessage = modelReady ? "Translating..." : "Loading model..."
        errorMessage = nil

        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            let start = DispatchTime.now()
            let record = await self.engine.generate(prompt: prompt) { token in
                Task { @MainActor in
                    self.appendToken(token, messageID: assistant.id, in: sessionID)
                }
            }
            await MainActor.run {
                self.loadingMessage = "Ready"
                self.isGenerating = false
                self.diagnostics = DiagnosticsSnapshot(
                    lastRun: Date(),
                    lastDurationMs: record.durationMs,
                    lastTokenCount: record.tokenCount,
                    lastRawLog: record.text,
                    lastStopReason: record.stopped ? "stopped" : (record.error != nil ? "error" : "completed")
                )
                if let err = record.error { self.errorMessage = err }
                self.persistSessions()
            }
        }
    }

    func stopGeneration() {
        Task { await engine.requestStop() }
        isGenerating = false
        loadingMessage = "Stopped"
    }

    func prepareModel() async {
        loadingMessage = "Preparing model..."
        let ok = await prepManager.prepare(using: engine) { progress in
            self.preparationProgress = progress
        }
        await MainActor.run {
            self.markModelReady(ok)
            self.loadingMessage = ok ? "Model ready" : "Preparation failed"
            self.preparationProgress = 0
        }
    }

    func messages(for id: UUID) -> [Message] {
        sessions.first(where: { $0.id == id })?.messages ?? []
    }

    private func appendMessage(_ message: Message, to sessionID: UUID) {
        guard let idx = sessions.firstIndex(where: { $0.id == sessionID }) else { return }
        sessions[idx].messages.append(message)
        sessions[idx].updatedAt = Date()
        persistSessions()
    }

    private func appendToken(_ token: String, messageID: UUID, in sessionID: UUID) {
        guard let sIdx = sessions.firstIndex(where: { $0.id == sessionID }),
              let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageID }) else { return }
        sessions[sIdx].messages[mIdx].content.append(token)
        sessions[sIdx].updatedAt = Date()
    }

    private func persistSessions() {
        historyStore.save(sessions)
    }

    deinit {
        Task { await engine.destroy() }
    }
}

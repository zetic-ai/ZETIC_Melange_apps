import Foundation
import SwiftUI

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var sessions: [Session] = []
    @Published var currentSessionID: UUID?
    @Published var draft: String = ""
    @Published var isGenerating = false
    @Published var isModelLoading = false
    @Published var isDownloading = true
    @Published var downloadProgress: Float = 0.0
    @Published var initializationState: String = "Checking Model..."
    @Published var loadingMessage = "Ready"
    @Published var scrollTrigger: Int = 0
    @Published var networkStatus = "Unknown"
    @Published var diagnostics = DiagnosticsSnapshot(lastRun: nil, lastDurationMs: 0, lastTokenCount: 0, lastRawLog: "", lastStopReason: "idle")
    @Published var themeOverride: ColorScheme? = nil
    @Published var systemPrompt: String = SettingsStore.defaultPrompt
    @Published var errorMessage: String?

    private let sessionStore = SessionStore()
    private let settingsStore = SettingsStore()
    private let compat = ModelCompatibility()
    private var engine: ZeticChatEngine?

    func bootstrap() async {
        sessions = sessionStore.load()
        if currentSessionID == nil {
            if let first = sessions.first { currentSessionID = first.id } else { createSession() }
        }
        themeOverride = settingsStore.theme
        systemPrompt = settingsStore.systemPrompt

        isModelLoading = true
        isDownloading = true
        initializationState = "Checking Model..."
        loadingMessage = "Preparing model…"

        let newEngine = ZeticChatEngine()
        do {
            try await newEngine.initialize { [weak self] progress in
                Task { @MainActor in
                    self?.downloadProgress = progress
                    if progress > 0.0 {
                        self?.initializationState = "Downloading Model (\(Int(progress * 100))%)"
                    }
                }
            }
            engine = newEngine
        } catch {
            initializationState = "Model Error"
            errorMessage = "Model initialization failed: \(error.localizedDescription)"
        }

        isDownloading = false
        isModelLoading = false
        loadingMessage = "Ready"
    }

    func createSession() {
        guard !isGenerating else { return }
        let new = Session(id: UUID(), title: "New Session", updatedAt: Date(), messages: [])
        sessions.insert(new, at: 0)
        currentSessionID = new.id
        persist()
    }

    func selectSession(_ id: UUID) { currentSessionID = id }

    func sendMessage() {
        guard let sessionID = currentSessionID else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating, !isDownloading else { return }
        guard let engine else { errorMessage = "Model not initialized"; return }

        let userMsg = Message(id: UUID(), role: .user, content: text, timestamp: Date())
        draft = ""
        appendMessage(userMsg, to: sessionID)

        let assistant = Message(id: UUID(), role: .assistant, content: "", timestamp: Date())
        appendMessage(assistant, to: sessionID)

        let history = messages(for: sessionID)
        let prompt = compat.buildPrompt(systemPrompt: systemPrompt, history: Array(history.dropLast()), userInput: text)

        isGenerating = true
        loadingMessage = "Loading Model..."

        Task.detached { [weak self] in
            guard let self else { return }
            let start = Date()
            let record = await engine.generate(prompt: prompt) { token in
                Task { @MainActor in
                    self.updateAssistantMessage(assistant.id, contentAppend: token, in: sessionID)
                }
            }
            await MainActor.run {
                self.isGenerating = false
                self.loadingMessage = "Ready"
                self.diagnostics = DiagnosticsSnapshot(
                    lastRun: start,
                    lastDurationMs: record.durationMs,
                    lastTokenCount: record.tokenCount,
                    lastRawLog: record.text,
                    lastStopReason: record.error != nil ? "error" : (record.stopped ? "stopped" : "completed")
                )
                if let err = record.error { self.errorMessage = err }
                self.persist()
            }
        }
    }

    func stopGeneration() {
        if let engine {
            Task { await engine.requestStop() }
        }
        isGenerating = false
        loadingMessage = "Generation stopped"
    }

    func clearHistory() {
        stopGeneration()
        sessions = []
        sessionStore.clear()
        createSession()
    }

    func setTheme(_ scheme: ColorScheme?) {
        themeOverride = scheme
        settingsStore.setTheme(scheme)
    }

    func setSystemPrompt(_ prompt: String) {
        systemPrompt = prompt
        settingsStore.setPrompt(prompt)
    }

    func messages(for sessionID: UUID) -> [Message] {
        sessions.first(where: { $0.id == sessionID })?.messages ?? []
    }

    private func appendMessage(_ message: Message, to sessionID: UUID) {
        guard let idx = sessions.firstIndex(where: { $0.id == sessionID }) else { return }
        sessions[idx].messages.append(message)
        sessions[idx].updatedAt = Date()
        persist()
    }

    private func updateAssistantMessage(_ messageID: UUID, contentAppend: String, in sessionID: UUID) {
        guard let sIdx = sessions.firstIndex(where: { $0.id == sessionID }),
              let mIdx = sessions[sIdx].messages.firstIndex(where: { $0.id == messageID }) else { return }
        sessions[sIdx].messages[mIdx].content.append(contentAppend)
        sessions[sIdx].updatedAt = Date()
        scrollTrigger += 1
    }

    private func persist() { sessionStore.save(sessions) }

    deinit {
        let currentEngine = engine
        Task { await currentEngine?.destroy() }
    }
}

import SwiftUI

/// Root state for the container app's compose experience. Owns the input, the
/// selected task + its options, and the streamed result. Drives generation through
/// the warm `LLMService`.
@MainActor
final class AppModel: ObservableObject {
    // Input + task selection
    @Published var inputText = ""
    @Published var task: KeyboardTask = .rewrite
    @Published var tone: Tone = .professional
    @Published var stance: Stance = .agreeable
    @Published var targetLanguage: Language = .named("ko")

    // Result
    @Published var resultText = ""
    @Published var isGenerating = false
    @Published var hasResult = false
    @Published var errorMessage: String?
    @Published var didApply = false
    /// True when this run was started from the keyboard handoff (drives the
    /// "switch back and paste" guidance and auto-copy).
    @Published var fromKeyboard = false

    let llm = LLMService.shared

    /// Set when the compose screen was opened from a keyboard handoff, so the
    /// finished result can be written back to the App Group.
    private var pendingRequestID: UUID?
    private var genTask: Task<Void, Never>?

    var canGenerate: Bool {
        !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating
    }

    // MARK: Generation

    func run() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }

        genTask?.cancel()
        resultText = ""
        errorMessage = nil
        didApply = false
        hasResult = true
        isGenerating = true

        let prompt = Prompts.build(
            task: task, text: text, tone: tone, stance: stance,
            targetLanguage: targetLanguage.englishName
        )
        let maxTokens = Prompts.maxTokens(for: task)

        genTask = Task { [weak self] in
            guard let self else { return }
            await self.llm.ensureLoaded()
            guard self.llm.isModelReady else {
                self.isGenerating = false
                self.errorMessage = self.llm.loadError ?? "The model isn't ready yet."
                return
            }
            do {
                let final = try await self.llm.generateSanitized(
                    prompt: prompt, maxTokens: maxTokens
                ) { partial in
                    self.resultText = partial
                }
                self.resultText = final
                self.isGenerating = false
                self.publishResultIfNeeded(final)
            } catch is CancellationError {
                self.isGenerating = false
            } catch {
                self.isGenerating = false
                self.errorMessage = error.localizedDescription
            }
        }
    }

    func retake() { run() }

    func cancel() {
        genTask?.cancel()
        isGenerating = false
    }

    /// Copies the result to the pasteboard and writes it back for the keyboard.
    func apply() {
        guard !resultText.isEmpty else { return }
        KeyboardBridge.copyToPasteboard(resultText)
        publishResultIfNeeded(resultText)
        didApply = true
    }

    private func publishResultIfNeeded(_ text: String) {
        guard let id = pendingRequestID, !text.isEmpty else { return }
        KeyboardBridge.publishResult(requestID: id, text: text)
        // Auto-copy so the user can immediately paste back in the host app.
        KeyboardBridge.copyToPasteboard(text)
    }

    // MARK: Keyboard handoff

    /// Handles a `cherrypad://process?id=…` deep link from the keyboard.
    func handleDeepLink(_ url: URL) {
        guard DeepLink.requestID(from: url) != nil else { return }
        consumePendingRequest()
    }

    /// Fallback path: pull the latest pending request from the App Group (used when
    /// the app returns to the foreground in case the URL was dropped).
    func consumePendingRequest() {
        guard let req = HandoffStore.readRequest() else { return }
        pendingRequestID = req.id
        fromKeyboard = true
        inputText = req.text
        task = req.task
        if let t = req.tone { tone = t }
        if let s = req.stance { stance = s }
        if let name = req.targetLanguage, let lang = Language.matching(englishName: name) {
            targetLanguage = lang
        }
        HandoffStore.clearRequest()
        run()
    }
}

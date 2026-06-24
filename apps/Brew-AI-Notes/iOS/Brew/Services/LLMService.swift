import Foundation
import OSLog
import os

enum LLMError: LocalizedError {
    case notReady
    case missingKey
    var errorDescription: String? {
        switch self {
        case .notReady: return "The on-device model isn't ready yet."
        case .missingKey: return "No AI model key configured."
        }
    }
}

/// Single owner of the local model. Picks the right `LLMEngine` for the build
/// environment (real ZeticMLange on device, a stub in the Simulator) and
/// funnels all access through one serial queue so the single generation context
/// is never used concurrently. Published state drives the download UI.
@MainActor
final class LLMService: ObservableObject {
    static let shared = LLMService()

    /// Lifecycle of on-device model preparation. `downloading` carries real
    /// file-download progress; `preparing` covers initialization (loading and
    /// compiling weights), which the SDK reports no progress for — so the UI
    /// shows an indeterminate state there instead of a frozen 0%.
    enum Phase: Equatable {
        case idle
        case downloading(Double)   // 0...1 file download
        case preparing             // files present; initializing the model
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    /// True once the model is loaded and able to generate.
    var isModelReady: Bool { phase == .ready }

    /// The failure message when preparation failed, else nil.
    var loadError: String? {
        if case .failed(let message) = phase { return message }
        return nil
    }

    /// Real download progress while a file download is in flight, else nil. Kept
    /// for surfaces that only render a percentage bar.
    var downloadProgress: Double? {
        if case .downloading(let value) = phase { return value }
        return nil
    }

    /// Preparation has settled — either ready, or failed in a way the user can
    /// retry later. Meeting capture is gated on this: a meeting can't start while
    /// the model is still preparing, but a model failure never blocks recording
    /// (audio still captures and transcribes; the AI note is generated later).
    var preparationResolved: Bool {
        switch phase {
        case .ready, .failed: return true
        case .idle, .downloading, .preparing: return false
        }
    }

    private let queue = DispatchQueue(label: "com.brew.llm", qos: .userInitiated)
    private let engine: LLMEngine
    private var loadTask: Task<Void, Never>?

    private init() {
        #if targetEnvironment(simulator)
        engine = StubLLMEngine()
        #else
        let key = "YOUR_MLANGE_KEY"
        engine = ZeticLLMEngine(
            personalKey: key,
            modelName: "changgeun/gemma-4-E2B-it"
        )
        #endif
    }

    /// Downloads + initializes the model. Idempotent and safe to call from
    /// multiple places — concurrent callers await the same in-flight load, so
    /// preloading at launch and a later "Generate" tap share one download.
    func ensureLoaded() async {
        if isModelReady { return }
        if let task = loadTask { await task.value; return }
        let task = Task { [weak self] in
            guard let self else { return }
            await self.performLoad()
        }
        loadTask = task
        await task.value
        if loadError != nil { loadTask = nil } // allow retry after a failure
    }

    private func performLoad() async {
        // Start indeterminate: a cached model reports no download progress, so
        // showing 0% here would look frozen. A percentage is surfaced only once a
        // real, in-flight download value arrives.
        phase = .preparing
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                guard let self else { cont.resume(); return }
                do {
                    try self.engine.load { progress in
                        Task { @MainActor in
                            // Mid-download → show the percentage. At 0 or 1 the
                            // SDK is downloading nothing / initializing, which has
                            // no progress signal, so stay indeterminate.
                            self.phase = (progress > 0 && progress < 1)
                                ? .downloading(progress)
                                : .preparing
                        }
                    }
                    // Resume only after the terminal state is applied, so callers
                    // awaiting ensureLoaded() observe a settled phase.
                    Task { @MainActor in self.phase = .ready; cont.resume() }
                } catch {
                    let message = error.localizedDescription
                    Task { @MainActor in self.phase = .failed(message); cont.resume() }
                }
            }
        }
    }

    /// Streams generated tokens for a prompt. The blocking token loop runs on
    /// the serial queue; tokens are yielded as they arrive. The stream stops
    /// early when the consuming Task is cancelled (e.g. the chat sheet is
    /// dismissed) or `maxTokens` is reached, so the CPU isn't burned on output
    /// nobody will see.
    func generate(prompt: String, maxTokens: Int = 1024) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let cancelled = OSAllocatedUnfairLock(initialState: false)
            continuation.onTermination = { reason in
                if case .cancelled = reason {
                    cancelled.withLock { $0 = true }
                }
            }
            queue.async { [weak self] in
                guard let self else {
                    continuation.finish(throwing: LLMError.notReady)
                    return
                }
                do {
                    let promptTokens = try self.engine.startGeneration(prompt: prompt)
                    defer { self.engine.stopGeneration() }
                    Self.log.info("Prompt tokens: \(promptTokens)")
                    var generated = 0
                    while true {
                        if cancelled.withLock({ $0 }) {
                            break
                        }
                        let result = self.engine.nextToken()
                        if !result.token.isEmpty {
                            continuation.yield(result.token)
                        }
                        if result.isFinished { break }
                        generated += 1
                        if generated >= maxTokens { break }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private static let log = Logger(subsystem: "com.brew.app", category: "llm")

    /// Streams a generation with sanitized, UI-rate-limited updates: `onUpdate`
    /// fires with the cleaned text on the first token (fast perceived
    /// response), then at most every 100ms — sanitizing and publishing per
    /// token is O(n²) churn. Returns the final sanitized text.
    func generateSanitized(
        prompt: String,
        maxTokens: Int = 1024,
        onUpdate: (String) -> Void
    ) async throws -> String {
        var raw = ""
        var lastFlush: ContinuousClock.Instant?
        for try await token in generate(prompt: prompt, maxTokens: maxTokens) {
            raw += token
            let now = ContinuousClock.now
            if lastFlush == nil || lastFlush!.duration(to: now) >= .milliseconds(100) {
                onUpdate(LLMOutput.sanitize(raw))
                lastFlush = now
            }
        }
        return LLMOutput.sanitize(raw)
    }
}

import Foundation
import OSLog
import os

/// Single owner of the local model. Picks the right `LLMEngine` for the build
/// environment (real ZeticMLange on device, a stub in the Simulator) and funnels
/// all access through one serial queue so the single generation context is never
/// used concurrently. The model is loaded once at launch and kept WARM for the
/// whole session — only `cleanUp()` (KV reset) runs between requests, never a
/// reload — so a tapped action incurs zero load cost. Published state drives the
/// download UI.
@MainActor
final class LLMService: ObservableObject {
    static let shared = LLMService()

    /// Lifecycle of on-device model preparation. `downloading` carries real
    /// file-download progress; `preparing` covers initialization (loading and
    /// compiling weights), which the SDK reports no progress for.
    enum Phase: Equatable {
        case idle
        case downloading(Double)   // 0...1 file download
        case preparing             // files present; initializing the model
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    var isModelReady: Bool { phase == .ready }

    var loadError: String? {
        if case .failed(let message) = phase { return message }
        return nil
    }

    var downloadProgress: Double? {
        if case .downloading(let value) = phase { return value }
        return nil
    }

    private let queue = DispatchQueue(label: "ai.zetic.demo.CherryPad.llm", qos: .userInitiated)
    private var engine: LLMEngine
    private var loadTask: Task<Void, Never>?

    private init() {
        engine = Self.makeEngine()
    }

    private static func makeEngine() -> LLMEngine {
        #if targetEnvironment(simulator)
        return StubLLMEngine()
        #else
        return ZeticLLMEngine(
            personalKey: ZeticConfig.personalKey,
            modelName: ZeticConfig.modelName,
            accuracyMode: ZeticConfig.usesAccuracyMode
        )
        #endif
    }

    /// Downloads + initializes the model. Idempotent and safe to call from
    /// multiple places — concurrent callers await the same in-flight load, so
    /// preloading at launch and a later action share one load.
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
        phase = .preparing
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [weak self] in
                guard let self else { cont.resume(); return }
                do {
                    try self.engine.load { progress in
                        Task { @MainActor in
                            self.phase = (progress > 0 && progress < 1)
                                ? .downloading(progress)
                                : .preparing
                        }
                    }
                    Task { @MainActor in self.phase = .ready; cont.resume() }
                } catch {
                    let message = error.localizedDescription
                    Task { @MainActor in self.phase = .failed(message); cont.resume() }
                }
            }
        }
    }

    /// Streams generated tokens for a prompt. The blocking token loop runs on the
    /// serial queue; tokens are yielded as they arrive. Stops early when the
    /// consuming Task is cancelled (e.g. Retake) or `maxTokens` is reached.
    /// `defer { stopGeneration() }` guarantees every generation ends on a clean KV
    /// cache, keeping the model warm for the next request.
    func generate(prompt: String, maxTokens: Int) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let cancelled = OSAllocatedUnfairLock(initialState: false)
            continuation.onTermination = { reason in
                if case .cancelled = reason { cancelled.withLock { $0 = true } }
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
                        if cancelled.withLock({ $0 }) { break }
                        let result = self.engine.nextToken()
                        // The SDK signals end-of-generation with an empty token
                        // (matches the working Qwen3Chat app); isFinished alone is
                        // not always reliable.
                        if result.token.isEmpty { break }
                        continuation.yield(result.token)
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

    /// Streams a generation with sanitized, UI-rate-limited updates: `onUpdate`
    /// fires with cleaned text on the first token (fast perceived response), then
    /// at most every 100ms. Returns the final sanitized text.
    func generateSanitized(
        prompt: String,
        maxTokens: Int,
        onUpdate: @escaping (String) -> Void
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

    private static let log = Logger(subsystem: "ai.zetic.demo.CherryPad", category: "llm")
}

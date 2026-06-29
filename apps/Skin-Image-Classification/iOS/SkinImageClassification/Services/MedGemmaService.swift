import Foundation
import OSLog
import os
import ZeticMLange

/// Owns the on-device MedGemma model and performs all SDK calls. Not main-actor
/// isolated: every method here runs on `MedGemmaService`'s serial queue, which is
/// the synchronization point, so the model is only ever touched from one thread.
/// Marked `@unchecked Sendable` for that reason.
private final class MedGemmaEngine: @unchecked Sendable {
    private var model: ZeticMLangeLLMModel?
    private let log = Logger(subsystem: "ai.zetic.demo.SkinImageClassification", category: "llm")

    var isLoaded: Bool { model != nil }

    /// Build (download on first run) + initialize the model. `onProgress` reports 0…1.
    func load(onProgress: @escaping (Float) -> Void) throws {
        model = try Self.build(log: log, onDownload: onProgress)
    }

    private static func build(log: Logger, onDownload: ((Float) -> Void)? = nil) throws -> ZeticMLangeLLMModel {
        MemoryProbe.log("MedGemma: before init")
        log.info("MedGemma init START — name=\(AppConfig.Model.medGemma, privacy: .public) v\(AppConfig.Model.medGemmaVersion) mode=RUN_SPEED nCtx=\(AppConfig.LLM.contextTokens)")
        do {
            let m = try ZeticMLangeLLMModel(
                personalKey: AppConfig.personalKey,
                name: AppConfig.Model.medGemma,
                version: AppConfig.Model.medGemmaVersion,
                // RUN_SPEED matches the repo's working LLM app (Brew-AI-Notes); RUN_AUTO's
                // init-time backend selection was crashing with EXC_BAD_ACCESS.
                modelMode: LLMModelMode.RUN_SPEED,
                initOption: LLMInitOption(kvCacheCleanupPolicy: .CLEAN_UP_ON_FULL, nCtx: AppConfig.LLM.contextTokens),
                onDownload: { p in onDownload?(p) }
            )
            log.info("MedGemma init DONE")
            MemoryProbe.log("MedGemma: after init")
            return m
        } catch {
            log.error("MedGemma init THREW: \(error.localizedDescription, privacy: .public)")
            throw error
        }
    }

    /// Start generating `prompt`. If `run()` throws (e.g. a corrupted context), the
    /// model is force-rebuilt once and retried — so we don't reload multi-GB weights
    /// between every analysis on the happy path.
    func startGeneration(_ prompt: String) throws {
        guard let model else { throw SimpleError("Model not ready") }
        do {
            _ = try model.run(prompt)
        } catch {
            log.error("run() failed, rebuilding once: \(error.localizedDescription, privacy: .public)")
            model.forceDeinit()
            self.model = nil
            let fresh = try Self.build(log: log)
            self.model = fresh
            _ = try fresh.run(prompt)
        }
    }

    func nextToken() -> LLMNextTokenResult { model?.waitForNextToken() ?? .init(token: "", generatedTokens: 0, code: 0) }

    func finishGeneration() { try? model?.cleanUp() }
}

/// Single owner of the on-device MedGemma LLM. `@MainActor` for its published load
/// state; all SDK work is delegated to a nonisolated `MedGemmaEngine` on one serial
/// queue (the generation context is not concurrency-safe). Tokens stream as they arrive.
@MainActor
final class MedGemmaService: ObservableObject {

    enum Phase: Equatable {
        case idle
        case downloading(Double)   // 0…1 file download
        case preparing             // files present; compiling/initializing
        case ready
        case failed(String)
    }

    @Published private(set) var phase: Phase = .idle

    var isReady: Bool { phase == .ready }
    var loadError: String? { if case .failed(let m) = phase { return m }; return nil }

    /// Bridge to the shared `LoadPhase` so the download UI can render both models uniformly.
    var loadPhase: LoadPhase {
        switch phase {
        case .idle: return .idle
        case .downloading(let p): return .downloading(p)
        case .preparing: return .preparing
        case .ready: return .ready
        case .failed(let m): return .failed(m)
        }
    }

    private let queue = DispatchQueue(label: "ai.zetic.medgemma.llm", qos: .userInitiated)
    private let engine = MedGemmaEngine()
    private var loadTask: Task<Void, Never>?

    // MARK: Loading

    /// Idempotent download + initialize. Concurrent callers await the same load.
    func ensureLoaded() async {
        if isReady { return }
        if let task = loadTask { await task.value; return }
        let task = Task { [weak self] in
            guard let self else { return }
            await self.performLoad()
        }
        loadTask = task
        await task.value
        if loadError != nil { loadTask = nil }   // allow retry after failure
    }

    private func performLoad() async {
        phase = .preparing
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async { [engine] in
                do {
                    try engine.load { progress in
                        Task { @MainActor [weak self] in
                            self?.phase = (progress > 0 && progress < 1) ? .downloading(Double(progress)) : .preparing
                        }
                    }
                    Task { @MainActor [weak self] in
                        MemoryProbe.log("MedGemma loaded")
                        self?.phase = .ready
                        cont.resume()
                    }
                } catch {
                    let message = error.localizedDescription
                    Task { @MainActor [weak self] in
                        self?.phase = .failed(message.isEmpty ? "Couldn't load the medical model." : message)
                        cont.resume()
                    }
                }
            }
        }
    }

    // MARK: Generation

    /// Streams generated tokens for `prompt`. The blocking token loop runs on the
    /// serial queue; tokens are yielded as they arrive, stopping early if the consumer
    /// is cancelled or `maxTokens` is hit.
    func generate(prompt: String, maxTokens: Int = AppConfig.LLM.maxTokens) -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream { continuation in
            let cancelled = OSAllocatedUnfairLock(initialState: false)
            continuation.onTermination = { reason in
                if case .cancelled = reason { cancelled.withLock { $0 = true } }
            }
            queue.async { [engine] in
                do {
                    try engine.startGeneration(prompt)
                    defer { engine.finishGeneration() }
                    var generated = 0
                    while true {
                        if cancelled.withLock({ $0 }) { break }
                        let result = engine.nextToken()
                        if result.generatedTokens == 0 { break }   // per Melange deployment contract
                        if !result.token.isEmpty { continuation.yield(result.token) }
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

    /// Streams with sanitized, UI-rate-limited updates: `onUpdate` fires with cleaned
    /// text on the first token, then at most every 100ms (sanitizing per token is
    /// O(n²) churn). Returns the final sanitized text.
    func generateSanitized(prompt: String,
                           maxTokens: Int = AppConfig.LLM.maxTokens,
                           onUpdate: @escaping (String) -> Void) async throws -> String {
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

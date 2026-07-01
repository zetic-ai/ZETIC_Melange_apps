import Foundation

/// Runs the model INSIDE the keyboard extension so AI actions work without opening
/// the app. iOS caps keyboard extensions at ~48–60MB of dirty working memory, so
/// this is tuned hard for that: LFM2.5-350M (weights are mmap'd, mostly uncounted),
/// a small context window (nCtx=512 → tiny KV cache), a hard output-token cap, and
/// truncated input. Those bound the dirty KV pages that would otherwise grow per
/// generated token and get the extension jetsam-killed.
final class KeyboardLLM {
    static let shared = KeyboardLLM()

    enum Status { case downloading(Double), thinking }

    /// Keyboard-only limits (smaller than the app's) to survive the memory ceiling.
    private static let nCtx = 512
    private static let maxOutputTokens = 64
    private static let maxInputChars = 400

    private let queue = DispatchQueue(label: "ai.zetic.demo.CherryPad.kbllm", qos: .userInitiated)
    private var engine: LLMEngine?
    private var loaded = false
    private var busy = false

    private init() {}

    private func makeEngine() -> LLMEngine {
        #if targetEnvironment(simulator)
        return StubLLMEngine()
        #else
        return ZeticLLMEngine(
            personalKey: ZeticConfig.personalKey,
            modelName: ZeticConfig.modelName,
            accuracyMode: ZeticConfig.usesAccuracyMode,
            nCtx: Self.nCtx
        )
        #endif
    }

    var isBusy: Bool { busy }

    /// Frees the model (e.g. on a memory warning) so the next action reloads instead
    /// of dying mid-use.
    func unload() {
        queue.async { [weak self] in
            guard let self, !self.busy else { return }
            self.engine?.unload()
            self.engine = nil
            self.loaded = false
        }
    }

    func generate(task: KeyboardTask,
                  text rawText: String,
                  tone: Tone?,
                  stance: Stance?,
                  targetLanguage: String?,
                  onStatus: @escaping (Status) -> Void,
                  completion: @escaping (Result<String, Error>) -> Void) {
        queue.async { [weak self] in
            guard let self else { return }
            self.busy = true
            defer { self.busy = false }

            if self.engine == nil { self.engine = self.makeEngine() }
            guard let engine = self.engine else {
                completion(.failure(LLMError.notReady)); return
            }
            if !self.loaded {
                do {
                    try engine.load { progress in onStatus(.downloading(progress)) }
                    self.loaded = true
                } catch {
                    self.engine = nil
                    completion(.failure(error)); return
                }
            }

            onStatus(.thinking)
            let text = String(rawText.prefix(Self.maxInputChars))
            let prompt = Prompts.build(task: task, text: text, tone: tone,
                                       stance: stance, targetLanguage: targetLanguage)
            do {
                _ = try engine.startGeneration(prompt: prompt)
                defer { engine.stopGeneration() }
                var raw = ""
                var n = 0
                while n < Self.maxOutputTokens {
                    let (token, finished) = engine.nextToken()
                    if token.isEmpty { break }
                    raw += token
                    if finished { break }
                    n += 1
                }
                completion(.success(LLMOutput.sanitize(raw)))
            } catch {
                completion(.failure(error))
            }
        }
    }
}

import Foundation
import ZeticMLange

actor ZeticChatEngine {
    private var model: ZeticMLangeLLMModel?
    private var stopRequested = false

    static let modelId = "Steve/kanana_1_5__2_1b_instruct"
    static let personalKey = "YOUR_MLANGE_KEY"

    var isInitialized: Bool { model != nil }

    init() {}

    func initialize(onDownloadProgress: @escaping @Sendable (Float) -> Void) throws {
        guard model == nil else { return }
        model = try ZeticMLangeLLMModel(personalKey: Self.personalKey, name: Self.modelId) { progress in
            onDownloadProgress(progress)
        }
    }

    func generate(prompt: String, onToken: @Sendable @escaping (String) -> Void) async -> GenerationRecord {
        guard let model = model else {
            return GenerationRecord(text: "", tokenCount: 0, durationMs: 0, stopped: false, error: "Model not initialized")
        }
        try? model.cleanUp()
        stopRequested = false

        let start = DispatchTime.now()
        var text = ""
        var tokenCount = 0
        var err: String?

        do {
            _ = try model.run(prompt)
            while true {
                if stopRequested { break }
                let result = model.waitForNextToken()
                if result.generatedTokens == 0 { break }
                if !result.token.isEmpty {
                    text.append(result.token)
                    tokenCount += result.generatedTokens
                    await MainActor.run { onToken(result.token) }
                }
                try Task.checkCancellation()
            }
        } catch {
            err = error.localizedDescription
        }

        try? model.cleanUp()
        let durationMs = Int((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000)
        return GenerationRecord(text: text, tokenCount: tokenCount, durationMs: durationMs, stopped: stopRequested, error: err)
    }

    func requestStop() {
        stopRequested = true
        try? model?.cleanUp()
    }

    func destroy() {
        stopRequested = true
        try? model?.cleanUp()
    }

    static func maskedKey() -> String {
        let k = personalKey
        guard k.count >= 8 else { return "****" }
        return k.prefix(4) + "****" + k.suffix(4)
    }
}

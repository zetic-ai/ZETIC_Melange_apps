import Foundation
import ZeticMLange

actor TranslationEngine {
    private let model: ZeticMLangeLLMModel
    private var stopRequested = false

    private static let modelId = "vaibhav-zetic/tencent_HY-MT"
    private static let personalKey = "debug_b07fbc1d4e1145549fcc94bdf319a858"

    init() {
        model = try! ZeticMLangeLLMModel(personalKey: Self.personalKey, name: Self.modelId)
    }

    func generate(prompt: String, onToken: @Sendable @escaping (String) -> Void) async -> GenerationRecord {
        try? model.cleanUp()
        stopRequested = false
        var text = ""
        var tokenCount = 0
        var err: String?
        let start = DispatchTime.now()

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
        try? model.cleanUp()
    }

    func warmup() async -> Bool {
        try? model.cleanUp()
        stopRequested = false
        do {
            _ = try model.run("prepare")
            while true {
                let result = model.waitForNextToken()
                if result.generatedTokens == 0 { break }
                try Task.checkCancellation()
            }
        } catch {
            try? model.cleanUp()
            return false
        }
        try? model.cleanUp()
        return true
    }

    func destroy() {
        stopRequested = true
        try? model.cleanUp()
    }

    static func maskedKey() -> String {
        let k = personalKey
        guard k.count >= 8 else { return "****" }
        return k.prefix(4) + "****" + k.suffix(4)
    }

    static var modelName: String { modelId }
}

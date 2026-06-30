#if !targetEnvironment(simulator)
import Foundation
import OSLog
import ZeticMLange

/// Real on-device engine backed by ZeticMLange. Compiled only for physical
/// devices — the ZeticMLange package ships no Simulator slice.
///
/// We use `ZeticMLangeLLMModel` (not the generic raw-tensor `ZeticMLangeModel`
/// shown in the dashboard snippet): it tokenizes internally and exposes a
/// streaming token API, which is what the near-instant keyboard UX needs.
final class ZeticLLMEngine: LLMEngine {
    private var model: ZeticMLangeLLMModel?
    private let personalKey: String
    private let modelName: String
    private let accuracyMode: Bool
    private let log = Logger(subsystem: "ai.zetic.demo.CherryPad", category: "llm")

    init(personalKey: String, modelName: String, accuracyMode: Bool) {
        self.personalKey = personalKey
        self.modelName = modelName
        self.accuracyMode = accuracyMode
    }

    func load(onProgress: @escaping (Double) -> Void) throws {
        guard !personalKey.isEmpty else { throw LLMError.missingKey }
        // Qwen3-0.6B uses the SDK default mode (minimal init) — explicitly passing
        // a mode + custom LLMInitOption(nCtx:) made it degenerate on device. LFM2.5
        // uses RUN_ACCURACY per its dashboard recipe.
        if accuracyMode {
            model = try ZeticMLangeLLMModel(
                personalKey: personalKey,
                name: modelName,
                version: ZeticConfig.modelVersion,
                modelMode: LLMModelMode.RUN_ACCURACY,
                onDownload: { progress in onProgress(Double(progress)) }
            )
        } else {
            model = try ZeticMLangeLLMModel(
                personalKey: personalKey,
                name: modelName,
                version: ZeticConfig.modelVersion,
                onDownload: { progress in onProgress(Double(progress)) }
            )
        }
    }

    func startGeneration(prompt: String) throws -> Int {
        guard let model else { throw LLMError.notReady }
        let result = try model.run(prompt)
        return result.promptTokens
    }

    func nextToken() -> (token: String, isFinished: Bool) {
        guard let model else { return ("", true) }
        let result = model.waitForNextToken()
        return (result.token, result.isFinished)
    }

    func stopGeneration() {
        // Resets the KV cache only — keeps the weights warm for the next request.
        do {
            try model?.cleanUp()
        } catch {
            log.error("stopGeneration cleanUp failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    func unload() {
        model?.forceDeinit()
        model = nil
    }
}
#endif

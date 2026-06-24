#if canImport(ZeticMLange)
import Foundation
import ZeticMLange

/// Real on-device translator backed by the ZETIC.ai Melange SDK (`ZeticMLange` 1.6.0).
///
/// This whole file compiles only when the `ZeticMLange` package is linked (device target).
/// In the simulator/preview target the package is absent, so `canImport` is false and the
/// app falls back to `MockTranslator`.
final class ZeticTranslator: Translator {
    private var model: ZeticMLangeLLMModel?

    func load(onProgress: @escaping (Double) -> Void) throws {
        // Release any previously loaded instance before re-init (e.g. a retry after a
        // failed/cancelled load) so we never leak the old native model.
        model?.forceDeinit()
        model = nil
        // Blocking native init; first run downloads the model (progress 0...1).
        model = try ZeticMLangeLLMModel(
            personalKey: ZeticConfig.personalKey,
            name: ZeticConfig.modelName,
            version: ZeticConfig.modelVersion,
            modelMode: LLMModelMode.RUN_AUTO,
            onDownload: { progress in onProgress(Double(progress)) }
        )
    }

    func generate(prompt: String, onToken: (String) -> Bool) throws {
        guard let model else { throw TranslatorError.notLoaded }
        _ = try model.run(prompt)
        while true {
            let result = model.waitForNextToken() // blocks until the next token is ready
            if result.generatedTokens == 0 || result.isFinished { break }
            if !onToken(result.token) { break }    // cooperative cancel between tokens
        }
    }

    func reset() {
        try? model?.cleanUp() // reset KV-cache / conversation state, keep model loaded
    }

    func tearDown() {
        model?.forceDeinit()
        model = nil
    }
}
#endif

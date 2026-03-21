//
//  PromptGuardModel.swift
//  PromptGuard
//
//  Loads Zetic model and runs classification off the main thread.
//

import Foundation
@preconcurrency import ZeticMLange

private let modelName = "jathin-zetic/llama_prompt_guard_2"
private let modelVersion = 1
/// Max sequence length passed to input factory; must match export SEQ_LENGTH (128).
private let modelMaxTokens = 128

enum PromptGuardModelError: Error {
    case modelLoadFailed
    case runFailed(Error)
}

final class PromptGuardModel: ObservableObject, @unchecked Sendable {
    @Published private(set) var isLoaded = false
    @Published private(set) var isLoading = false
    @Published private(set) var lastError: String?
    @Published private(set) var lastLatencyMs: Double?
    @Published private(set) var lastRawOutput: String?
    @Published private(set) var downloadProgress: Int = 0

    private var model: ZeticMLangeModel?
    private let tokenizer = PromptGuardTokenizer()
    private let specStore = ModelInputSpecStore.shared

    /// True when tokenizer.json was loaded; false means fallback encoding is used (add tokenizer.json for best results).
    var isTokenizerLoaded: Bool { tokenizer.isLoaded }
    private let inferenceQueue = DispatchQueue(label: "promptguard.inference", qos: .userInitiated)

    init() {}

    func load() {
        guard model == nil, !isLoading else { return }
        DispatchQueue.main.async { [weak self] in
            self?.lastError = nil
            self?.isLoading = true
        }
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            do {
                let m = try ZeticMLangeModel(
                    personalKey: Config.personalKey,
                    name: modelName,
                    version: modelVersion
                )
                self.model = m
                DispatchQueue.main.async { [weak self] in
                    self?.isLoaded = true
                    self?.isLoading = false
                }
            } catch {
                let message = Self.userFriendlyModelLoadError(error)
                DispatchQueue.main.async { [weak self] in
                    self?.lastError = message
                    self?.isLoading = false
                }
            }
        }
    }

    private static func userFriendlyModelLoadError(_ error: Error) -> String {
        let text = error.localizedDescription
        if text.contains("404") || text.lowercased().contains("not found") ||
           text.lowercased().contains("not available for device") {
            return "Model not available for this device. Ensure the model has an iOS (CoreML) build published."
        }
        return text
    }

    func classify(userInput: String, agentOutput: String = "") async -> ClassificationResult? {
        guard let m = model else {
            await MainActor.run { [weak self] in self?.lastError = "Model not loaded" }
            return nil
        }
        return await withCheckedContinuation { (cont: CheckedContinuation<ClassificationResult?, Never>) in
            inferenceQueue.async { [weak self] in
                guard let self else { cont.resume(returning: nil); return }
                let spec = self.specStore.spec
                let prompt = spec.applied(userInput: userInput, agentOutput: agentOutput)

                PipelineLogger.logInput(prompt: prompt, userInput: userInput, agentOutput: agentOutput)

                self.tokenizer.ensureLoaded()
                let inputResult: TensorInputResult
                do {
                    inputResult = try ZeticTensorFactory.createInput(prompt: prompt, maxTokens: modelMaxTokens, tokenizer: self.tokenizer)
                } catch {
                    DispatchQueue.main.async {
                        self.lastError = "Input creation failed: \(error.localizedDescription)"
                        cont.resume(returning: nil)
                    }
                    return
                }

                PipelineLogger.logTokenization(tokenCount: inputResult.promptTokenCount, firstTokenIds: inputResult.firstTokenIdsForLogging, usedTokenizer: inputResult.usedTokenizer)

                let start = CFAbsoluteTimeGetCurrent()
                do {
                    let outputs = try m.run(inputs: inputResult.tensors)
                    let latencyMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    let outputData = outputs.map { $0.data }
                    let result = ClassificationResult.fromOutputs(outputData)

                    let scores = result.categoryScores
                    let topIdx = scores.enumerated().max(by: { $0.element < $1.element }).map(\.offset) ?? 0
                    let topScore = topIdx < scores.count ? scores[topIdx] : 0
                    // Llama Prompt Guard 2: index 0 = Benign, index 1 = Malicious (no S2/S3)
                    let categoryLabel = topIdx == 0 ? "Benign" : "Malicious"
                    PipelineLogger.logModelOutput(rawLogits: scores, topIndex: topIdx, topScore: topScore, categoryLabel: categoryLabel)

                    DispatchQueue.main.async {
                        self.lastLatencyMs = latencyMs
                        self.lastError = nil
                        self.lastRawOutput = result.rawOutputSummary
                        cont.resume(returning: result)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.lastLatencyMs = nil
                        self.lastError = error.localizedDescription
                        self.lastRawOutput = nil
                        cont.resume(returning: nil)
                    }
                }
            }
        }
    }
}

enum Config {
    static var personalKey: String {
        ProcessInfo.processInfo.environment["ZETIC_PERSONAL_KEY"] ?? "YOUR_PERSONAL_KEY"
    }
}

//
//  PromptGuardModel.swift
//  PromptGuard
//
//  Loads Zetic model and runs classification off the main thread.
//  Init/run pattern matches working samples (e.g. FaceEmotionRecognition, FaceDetection).
//

import Foundation
@preconcurrency import ZeticMLange

// Model name — must match what’s published on Melange dashboard (https://melange.zetic.ai)
private let modelName = "jathin-zetic/llama_prompt_guard_2"
private let modelVersion = 1

enum PromptGuardModelError: Error {
    case modelLoadFailed
    case runFailed(Error)
}

final class PromptGuardModel: ObservableObject, @unchecked Sendable {
    @Published private(set) var isLoaded = false
    @Published private(set) var lastError: String?
    @Published private(set) var lastLatencyMs: Double?
    @Published private(set) var lastRawOutput: String?

    private var model: ZeticMLangeModel?
    private let specStore = ModelInputSpecStore.shared
    private let inferenceQueue = DispatchQueue(label: "promptguard.inference", qos: .userInitiated)

    init() {}

    func load() {
        guard model == nil else { return }
        DispatchQueue.main.async { [weak self] in self?.lastError = nil }
        do {
            // (1) Load Zetic MLange model — use Config.personalKey (set via env or replace placeholder)
            let m = try ZeticMLangeModel(
                personalKey: Config.personalKey,
                name: modelName,
                version: modelVersion
            )
            model = m
            DispatchQueue.main.async { [weak self] in self?.isLoaded = true }
        } catch {
            let message = Self.userFriendlyModelLoadError(error)
            DispatchQueue.main.async { [weak self] in self?.lastError = message }
        }
    }

    /// Turns SDK errors (404, "not available for device") into a short message and next steps.
    private static func userFriendlyModelLoadError(_ error: Error) -> String {
        let text = error.localizedDescription
        if text.contains("404") || text.lowercased().contains("not found") ||
           text.lowercased().contains("not available for device") {
            return "Model not available for this device. On the Zetic Melange dashboard, ensure the model has an iOS (CoreML) build published and your key has access."
        }
        return text
    }

    /// Runs inference on a background queue; updates @Published and returns result on completion.
    func classify(userInput: String, agentOutput: String = "") async -> ClassificationResult? {
        let spec = specStore.spec
        let prompt = spec.applied(userInput: userInput, agentOutput: agentOutput)
        // (2) Prepare model inputs — [Tensor] from prompt
        guard let tensorInputs = try? ZeticTensorFactory.createInput(prompt: prompt, maxTokens: spec.maxTokens) else {
            await MainActor.run { [weak self] in self?.lastError = "Input creation failed" }
            return nil
        }
        guard let m = model else {
            await MainActor.run { [weak self] in self?.lastError = "Model not loaded" }
            return nil
        }
        return await withCheckedContinuation { cont in
            inferenceQueue.async {
                let start = CFAbsoluteTimeGetCurrent()
                do {
                    // (3) Run and get output tensors of the model
                    let outputs = try m.run(inputs: tensorInputs)
                    let latencyMs = (CFAbsoluteTimeGetCurrent() - start) * 1000
                    let outputData = outputs.map { $0.data }
                    let result = ClassificationResult.fromOutputs(outputData)
                    DispatchQueue.main.async { [weak self] in
                        self?.lastLatencyMs = latencyMs
                        self?.lastError = nil
                        self?.lastRawOutput = result.rawOutputSummary
                        cont.resume(returning: result)
                    }
                } catch {
                    DispatchQueue.main.async { [weak self] in
                        self?.lastLatencyMs = nil
                        self?.lastError = error.localizedDescription
                        self?.lastRawOutput = nil
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

//
//  PipelineLogger.swift
//  PromptGuard
//
//  Logs tokenization and classification pipeline for debugging (e.g. why one category always wins).
//  View in Xcode console when running the app.
//  Llama Prompt Guard 2: index 0 = Benign, index 1 = Malicious (see HF model card).
//

import Foundation

enum PipelineLogger {
    private static let prefix = "[PromptGuard]"

    static func logInput(prompt: String, userInput: String, agentOutput: String) {
        let u = userInput.count > 80 ? String(userInput.prefix(80)) + "…" : userInput
        let a = agentOutput.isEmpty ? "empty" : (agentOutput.count > 40 ? String(agentOutput.prefix(40)) + "…" : agentOutput)
        print("\(prefix) [1] Input – userInput: \"\(u)\" | agentOutput: \"\(a)\"")
        let p = prompt.count > 200 ? String(prompt.prefix(200)) + "…" : prompt
        print("\(prefix) [1] Full prompt: \"\(p)\"")
    }

    static func logTokenization(tokenCount: Int, firstTokenIds: [Int32], usedTokenizer: Bool) {
        let head = firstTokenIds.prefix(12).map { "\($0)" }.joined(separator: ", ")
        print("\(prefix) [2] Tokenization – usedTokenizer: \(usedTokenizer) | tokenCount: \(tokenCount) | first 12 ids: [\(head)]")
    }

    static func logModelOutput(rawLogits: [Float], topIndex: Int, topScore: Float, categoryLabel: String) {
        let logitsStr = rawLogits.prefix(11).map { String(format: "%.3f", $0) }.joined(separator: ", ")
        print("\(prefix) [3] Model output – raw logits (first 11): [\(logitsStr)] | count: \(rawLogits.count)")
        print("\(prefix) [3] Predicted – index: \(topIndex) | score: \(String(format: "%.4f", topScore)) | category: \(categoryLabel)")
        if rawLogits.count <= 11 {
            print("\(prefix) [3] All scores (S1–S\(rawLogits.count)): \(rawLogits.enumerated().map { "S\($0.offset + 1)=\(String(format: "%.3f", $0.element))" }.joined(separator: ", "))")
        }
    }
}

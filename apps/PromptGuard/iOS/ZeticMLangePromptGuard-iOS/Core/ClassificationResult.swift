//
//  ClassificationResult.swift
//  PromptGuard
//

import Foundation

/// Parsed classification. Llama Prompt Guard 2 outputs 2 logits: Benign (index 0), Malicious (index 1).
struct ClassificationResult {
    /// Raw logits
    var categoryScores: [Float]
    var rawOutputData: Data?
    var rawOutputSummary: String

    /// Binary result per model card: "Benign" or "Malicious".
    var binaryLabel: String { isMalicious ? "Malicious" : "Benign" }
    /// Score of the winning class (Benign or Malicious logit).
    var binaryScore: Float { isMalicious ? (maliciousScore) : (benignScore) }
    var isMalicious: Bool { (maliciousScore) > (benignScore) }
    var benignScore: Float { categoryScores.count > 0 ? categoryScores[0] : 0 }
    var maliciousScore: Float { categoryScores.count > 1 ? categoryScores[1] : 0 }

    init(categoryScores: [Float], rawOutputData: Data? = nil, rawOutputSummary: String = "") {
        self.categoryScores = categoryScores
        self.rawOutputData = rawOutputData
        self.rawOutputSummary = rawOutputSummary
    }

    /// First tensor = logits (float). Llama Prompt Guard 2: 2 classes (Benign=0, Malicious=1).
    static func fromOutputs(_ outputs: [Data]) -> ClassificationResult {
        guard let first = outputs.first, first.count >= MemoryLayout<Float>.size else {
            return ClassificationResult(
                categoryScores: [0, 0],
                rawOutputSummary: "No output"
            )
        }
        let floatCount = first.count / MemoryLayout<Float>.size
        let floats: [Float] = first.withUnsafeBytes { ptr in
            guard let base = ptr.baseAddress?.assumingMemoryBound(to: Float.self) else { return [] }
            return Array(UnsafeBufferPointer(start: base, count: floatCount))
        }
        return fromFloats(floats, rawData: first)
    }

    private static func fromFloats(_ floats: [Float], rawData: Data) -> ClassificationResult {
        let count = min(11, max(2, floats.count))
        var scores = Array(floats.prefix(count))
        if scores.count < 2 { scores.append(contentsOf: repeatElement(Float(0), count: 2 - scores.count)) }
        if scores.count < 11 { scores.append(contentsOf: repeatElement(Float(0), count: 11 - scores.count)) }
        let summary = floats.prefix(20).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        return ClassificationResult(
            categoryScores: scores,
            rawOutputData: rawData,
            rawOutputSummary: "[\(summary)]… count=\(floats.count)"
        )
    }
}

//
//  ClassificationResult.swift
//  PromptGuard
//

import Foundation

/// Parsed classification: per-category score and optional raw output for Diagnostics.
struct ClassificationResult {
    /// Score per harm category (S1–S11). Order matches HarmCategory.allCases.
    var categoryScores: [Float]
    /// Raw output bytes (first output tensor) for Diagnostics viewer.
    var rawOutputData: Data?
    /// Human-readable raw summary (e.g. first N values).
    var rawOutputSummary: String

    init(categoryScores: [Float], rawOutputData: Data? = nil, rawOutputSummary: String = "") {
        self.categoryScores = categoryScores
        self.rawOutputData = rawOutputData
        self.rawOutputSummary = rawOutputSummary
    }

    /// Interpret model output [Data]. Assumes first tensor is float array; length ≥ 11 → first 11 as S1–S11; otherwise pad/truncate.
    static func fromOutputs(_ outputs: [Data]) -> ClassificationResult {
        guard let first = outputs.first, first.count >= MemoryLayout<Float>.size else {
            return ClassificationResult(
                categoryScores: Array(repeating: 0, count: 11),
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
        let count = min(11, max(0, floats.count))
        var scores = Array(floats.prefix(count))
        if scores.count < 11 {
            scores.append(contentsOf: repeatElement(Float(0), count: 11 - scores.count))
        }
        let summary = floats.prefix(20).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        return ClassificationResult(
            categoryScores: scores,
            rawOutputData: rawData,
            rawOutputSummary: "[\(summary)]… count=\(floats.count)"
        )
    }
}

import Foundation

/// One ranked (class, probability) pair from the classifier's softmax output.
struct ClassScore: Identifiable {
    let skinClass: SkinClass
    let probability: Float
    var id: Int { skinClass.rawValue }
}

/// The classifier's verdict for a single photo: the full softmax distribution
/// (ranked, highest first) plus inference latency.
struct Classification {
    /// All 7 classes ranked by probability, highest first.
    let ranked: [ClassScore]
    /// On-device inference time in milliseconds.
    let latencyMs: Double

    var top: ClassScore { ranked[0] }
    var topClass: SkinClass { top.skinClass }
    var confidence: Float { top.probability }

    /// Runner-up classes (everything below the top), for the distribution bars.
    var runnersUp: [ClassScore] { Array(ranked.dropFirst()) }

    /// Below this the model is effectively guessing — UI and MedGemma both soften.
    var isLowConfidence: Bool { confidence < 0.60 }

    /// Build from raw logits in `SkinClass.allCases` order.
    init(logits: [Float], latencyMs: Double) {
        let probs = Self.softmax(logits)
        ranked = SkinClass.allCases
            .map { ClassScore(skinClass: $0, probability: probs[safe: $0.rawValue] ?? 0) }
            .sorted { $0.probability > $1.probability }
        self.latencyMs = latencyMs
    }

    private static func softmax(_ x: [Float]) -> [Float] {
        guard let m = x.max() else { return x }
        let e = x.map { Foundation.exp($0 - m) }
        let sum = e.reduce(0, +)
        return sum > 0 ? e.map { $0 / sum } : e
    }
}

private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

import Foundation

/// One question + its on-device transcript, for the Transcript tab.
struct QAPair: Identifiable, Equatable {
    let id = UUID()
    let question: String
    let answer: String
}

/// The explainable, Aiberry-style readout produced at the end of a check-in.
/// Everything here is derived by a transparent rule (see `FusionEngine`) — there
/// is no trained scoring head, and it is **not** a diagnostic instrument.
struct ScreeningReport: Equatable {
    let wellbeing: Int          // 0…100 composite (higher = brighter affect)
    let band: String            // descriptive label for the gauge
    let mood: Int               // 0…100 from valence
    let energy: Int             // 0…100 from arousal
    let rateOfSpeech: Int       // 0…100 from voiced fraction
    let fused: [EmotionScore]   // blended face+voice distribution, ranked
    let faceTop: EmotionScore?
    let voiceTop: EmotionScore?
    let drivers: [String: [String]]   // dimension → top contributing emotion labels
    let confidence: Float       // 0…1 evidence quality
    let faceFrames: Int
    let transcript: [QAPair]
}

/// Transparent late-fusion of the face + voice emotion distributions into
/// Aiberry-style sub-dimensions and a composite well-being score. Pure functions,
/// unit-testable without a device.
enum FusionEngine {

    /// - Parameters:
    ///   - face: probabilities in `AppConfig.emotionLabels` order (empty if no face seen).
    ///   - faceFrames: number of good face frames behind `face`.
    ///   - voice: probabilities in `AppConfig.emotionLabels` order (empty if voice failed).
    ///   - voicedFraction: speaking time / total time, 0…1.
    static func fuse(face: [Float], faceFrames: Int,
                     voice: [Float], voicedFraction: Float,
                     transcript: [QAPair]) -> ScreeningReport {
        let labels = AppConfig.emotionLabels
        let n = labels.count

        let hasFace = face.count == n && faceFrames > 0
        let hasVoice = voice.count == n

        // Face/voice blend weight: trust face more with more good frames, but never
        // fully drop either present modality.
        var wFace: Float
        if hasFace && hasVoice {
            let raw = Float(faceFrames) / Float(AppConfig.Face.targetFrames)
            wFace = min(AppConfig.Fusion.faceWeightMax,
                        max(AppConfig.Fusion.faceWeightMin, raw))
        } else if hasFace {
            wFace = 1
        } else {
            wFace = 0
        }
        let wVoice = 1 - wFace

        let faceVec = hasFace ? face : [Float](repeating: 0, count: n)
        let voiceVec = hasVoice ? voice : [Float](repeating: 0, count: n)
        var fused = (0..<n).map { wFace * faceVec[$0] + wVoice * voiceVec[$0] }
        let sum = fused.reduce(0, +)
        if sum > 0 { fused = fused.map { $0 / sum } }

        // Circumplex projection.
        var valence: Float = 0, arousal: Float = 0
        for i in 0..<n {
            valence += fused[i] * (AppConfig.Affect.valence[labels[i]] ?? 0)
            arousal += fused[i] * (AppConfig.Affect.arousal[labels[i]] ?? 0)
        }

        let mood = clamp100(50 + 50 * valence)
        let energy = clamp100(50 + 50 * arousal)
        let rateRaw = voicedFraction / AppConfig.Fusion.typicalVoicedFraction
        let rate = clamp100(100 * min(max(rateRaw, 0), 1))

        // Energy & speech-rate are best near a mid-band, not maximal: triangular
        // preference peaking at the ideal. Mood is monotonic (brighter = better).
        let energyScore = midBand(Float(energy), ideal: AppConfig.Fusion.energyIdeal)
        let rateScore = midBand(Float(rate), ideal: AppConfig.Fusion.rateIdeal)
        let wellbeing = clamp100(
            AppConfig.Fusion.moodWeight * Float(mood)
            + AppConfig.Fusion.energyWeight * energyScore
            + AppConfig.Fusion.rateWeight * rateScore)

        let ranked = zip(labels, fused)
            .map { EmotionScore(label: $0.0, probability: $0.1) }
            .sorted { $0.probability > $1.probability }

        let drivers = [
            "Mood": topContributors(fused, labels: labels, coeff: AppConfig.Affect.valence),
            "Energy": topContributors(fused, labels: labels, coeff: AppConfig.Affect.arousal),
        ]

        let faceConf = min(1, Float(faceFrames) / Float(AppConfig.Face.targetFrames))
        let voiceConf: Float = hasVoice ? 1 : 0
        let confidence = hasFace && hasVoice ? 0.5 * faceConf + 0.5 * voiceConf
                                             : max(faceConf, voiceConf)

        return ScreeningReport(
            wellbeing: wellbeing,
            band: band(for: wellbeing),
            mood: mood, energy: energy, rateOfSpeech: rate,
            fused: ranked,
            faceTop: hasFace ? rankedTop(faceVec, labels) : nil,
            voiceTop: hasVoice ? rankedTop(voiceVec, labels) : nil,
            drivers: drivers,
            confidence: confidence,
            faceFrames: faceFrames,
            transcript: transcript)
    }

    // MARK: helpers

    private static func clamp100(_ x: Float) -> Int { Int(min(100, max(0, x)).rounded()) }

    /// Triangular preference peaking (=100) at `ideal`, falling to 0 toward the edges.
    private static func midBand(_ x: Float, ideal: Float) -> Float {
        max(0, 100 - 2 * abs(x - ideal))
    }

    private static func band(for wellbeing: Int) -> String {
        switch wellbeing {
        case 75...:  return "Bright"
        case 55..<75: return "Steady"
        case 35..<55: return "Guarded"
        default:      return "Low"
        }
    }

    private static func topContributors(_ dist: [Float], labels: [String],
                                        coeff: [String: Float], top: Int = 2) -> [String] {
        zip(labels, dist)
            .map { ($0.0, $0.1 * abs(coeff[$0.0] ?? 0)) }
            .sorted { $0.1 > $1.1 }
            .prefix(top)
            .filter { $0.1 > 0 }
            .map { $0.0 }
    }

    private static func rankedTop(_ dist: [Float], _ labels: [String]) -> EmotionScore? {
        zip(labels, dist)
            .map { EmotionScore(label: $0.0, probability: $0.1) }
            .max { $0.probability < $1.probability }
    }
}

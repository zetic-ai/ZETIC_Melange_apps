package ai.zetic.aiberry.session

import ai.zetic.aiberry.emotion.EmotionScore

/** One question + the (optionally transcribed) spoken answer. */
data class QAPair(val question: String, val answer: String)

/**
 * The end-of-session multimodal readout shown on the Insights screen. 1:1 with
 * iOS-Aiberry's `ScreeningReport`. Explicitly **non-diagnostic**.
 */
data class ScreeningReport(
    val wellbeing: Int,                 // 0..100 composite
    val band: String,                   // "Bright" | "Steady" | "Guarded" | "Low"
    val mood: Int,                      // 0..100 from valence
    val energy: Int,                    // 0..100 from arousal
    val rateOfSpeech: Int,              // 0..100 from voiced fraction
    val fused: List<EmotionScore>,      // 7 emotions ranked by blended probability
    val faceTop: EmotionScore?,         // top emotion from face alone (or null)
    val voiceTop: EmotionScore?,        // top emotion from voice alone (or null)
    val drivers: Map<String, List<String>>, // "Mood"/"Energy" -> top contributing emotions
    val confidence: Float,              // 0..1 evidence quality
    val faceFrames: Int,                // count of good face frames
    val transcript: List<QAPair>,       // one per answered question
)

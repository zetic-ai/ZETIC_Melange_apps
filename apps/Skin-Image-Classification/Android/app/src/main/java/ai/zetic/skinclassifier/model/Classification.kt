package ai.zetic.skinclassifier.model

import ai.zetic.skinclassifier.core.AppConfig
import ai.zetic.skinclassifier.core.MelangeKit

/** One class with its softmax probability. */
data class ClassScore(val skinClass: SkinClass, val probability: Float)

/**
 * Result of running the classifier on one image: every class ranked by probability plus the
 * on-device latency. 1:1 with the iOS `Classification` struct.
 */
data class Classification(
    val ranked: List<ClassScore>,
    val latencyMs: Double,
) {
    val top: ClassScore get() = ranked.first()
    val topClass: SkinClass get() = top.skinClass
    val confidence: Float get() = top.probability
    val isLowConfidence: Boolean get() = confidence < AppConfig.LOW_CONFIDENCE

    companion object {
        /** Build from raw logits (length >= 7): softmax, then rank classes descending. */
        fun fromLogits(logits: FloatArray, latencyMs: Double): Classification {
            val probs = MelangeKit.softmax(logits.copyOf(SkinClass.ordered.size))
            val ranked = SkinClass.ordered
                .mapIndexed { i, c -> ClassScore(c, probs[i]) }
                .sortedByDescending { it.probability }
            return Classification(ranked, latencyMs)
        }
    }
}

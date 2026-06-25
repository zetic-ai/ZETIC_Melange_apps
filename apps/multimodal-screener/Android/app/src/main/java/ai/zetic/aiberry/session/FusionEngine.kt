package ai.zetic.aiberry.session

import ai.zetic.aiberry.core.AppConfig
import ai.zetic.aiberry.emotion.EmotionScore
import kotlin.math.abs
import kotlin.math.min

/**
 * Late multimodal fusion: blends the face- and voice-emotion distributions, projects the
 * blend onto Russell's circumplex (valence -> Mood, arousal -> Energy), folds in a
 * rate-of-speech proxy, and composes a single 0..100 well-being score with a band + drivers.
 *
 * This is an EXACT port of iOS-Aiberry's `FusionEngine.fuse(...)` — same weights, same
 * clamps, same mid-band preference — so Android and iOS yield identical reports for identical
 * inputs. Rule-based on purpose: explainable, no extra model, non-diagnostic.
 */
object FusionEngine {

    private val labels = AppConfig.emotionLabels // canonical 7-order

    fun fuse(
        face: FloatArray,        // 7 probs in canonical order, or empty
        faceFrames: Int,
        voice: FloatArray,       // 7 probs in canonical order, or empty
        voicedFraction: Float,
        transcript: List<QAPair>,
    ): ScreeningReport {
        val hasFace = face.size == 7 && faceFrames > 0
        val hasVoice = voice.size == 7

        // --- Face/voice blend weight: trust face proportionally to good frame count. ---
        val wFace: Float = when {
            hasFace && hasVoice -> {
                val raw = faceFrames.toFloat() / AppConfig.Face.TARGET_FRAMES.toFloat()
                raw.coerceIn(AppConfig.Fusion.FACE_WEIGHT_MIN, AppConfig.Fusion.FACE_WEIGHT_MAX)
            }
            hasFace -> 1f
            else -> 0f
        }
        val wVoice = 1f - wFace

        val faceVec = if (hasFace) face else FloatArray(7)
        val voiceVec = if (hasVoice) voice else FloatArray(7)

        var fused = FloatArray(7) { wFace * faceVec[it] + wVoice * voiceVec[it] }
        val sum = fused.sum()
        if (sum > 0f) fused = FloatArray(7) { fused[it] / sum }

        // --- Russell circumplex projection (weighted sums over the blend). ---
        var valence = 0f
        var arousal = 0f
        for (i in 0 until 7) {
            valence += fused[i] * (AppConfig.Affect.valence[labels[i]] ?: 0f)
            arousal += fused[i] * (AppConfig.Affect.arousal[labels[i]] ?: 0f)
        }

        // --- Sub-dimensions (0..100). ---
        val mood = clamp100(50f + 50f * valence)
        val energy = clamp100(50f + 50f * arousal)
        val rateRaw = voicedFraction / AppConfig.Fusion.TYPICAL_VOICED_FRACTION
        val rate = clamp100(100f * rateRaw.coerceIn(0f, 1f))

        // --- Mid-band preference: Energy & Rate are healthiest near an ideal, not maxed. ---
        val energyScore = midBand(energy.toFloat(), AppConfig.Fusion.ENERGY_IDEAL)
        val rateScore = midBand(rate.toFloat(), AppConfig.Fusion.RATE_IDEAL)

        val wellbeing = clamp100(
            AppConfig.Fusion.MOOD_WEIGHT * mood +
                AppConfig.Fusion.ENERGY_WEIGHT * energyScore +
                AppConfig.Fusion.RATE_WEIGHT * rateScore,
        )

        // --- Drivers: which emotions most move Mood (valence) / Energy (arousal). ---
        val drivers = mapOf(
            "Mood" to topContributors(fused, AppConfig.Affect.valence),
            "Energy" to topContributors(fused, AppConfig.Affect.arousal),
        )

        // --- Confidence from evidence quality. ---
        val faceConf = min(1f, faceFrames.toFloat() / AppConfig.Face.TARGET_FRAMES.toFloat())
        val voiceConf = if (hasVoice) 1f else 0f
        val confidence = if (hasFace && hasVoice) {
            0.5f * faceConf + 0.5f * voiceConf
        } else {
            maxOf(faceConf, voiceConf)
        }

        val ranked = labels.indices
            .map { EmotionScore(labels[it], fused[it]) }
            .sortedByDescending { it.probability }

        val faceTop = if (hasFace) {
            labels.indices.map { EmotionScore(labels[it], face[it]) }.maxByOrNull { it.probability }
        } else null
        val voiceTop = if (hasVoice) {
            labels.indices.map { EmotionScore(labels[it], voice[it]) }.maxByOrNull { it.probability }
        } else null

        return ScreeningReport(
            wellbeing = wellbeing,
            band = band(wellbeing),
            mood = mood,
            energy = energy,
            rateOfSpeech = rate,
            fused = ranked,
            faceTop = faceTop,
            voiceTop = voiceTop,
            drivers = drivers,
            confidence = confidence,
            faceFrames = faceFrames,
            transcript = transcript,
        )
    }

    private fun clamp100(x: Float): Int = x.coerceIn(0f, 100f).toInt()

    /** Triangular score peaking at [ideal]: 100 at ideal, falling 2 points per unit away. */
    private fun midBand(x: Float, ideal: Float): Float = maxOf(0f, 100f - 2f * abs(x - ideal))

    private fun band(wellbeing: Int): String = when {
        wellbeing >= 75 -> "Bright"
        wellbeing >= 55 -> "Steady"
        wellbeing >= 35 -> "Guarded"
        else -> "Low"
    }

    /** Top up-to-2 emotions ranked by probability x |circumplex coefficient|. */
    private fun topContributors(dist: FloatArray, coeff: Map<String, Float>, top: Int = 2): List<String> =
        labels.indices
            .map { labels[it] to dist[it] * abs(coeff[labels[it]] ?: 0f) }
            .sortedByDescending { it.second }
            .take(top)
            .filter { it.second > 0f }
            .map { it.first }
}

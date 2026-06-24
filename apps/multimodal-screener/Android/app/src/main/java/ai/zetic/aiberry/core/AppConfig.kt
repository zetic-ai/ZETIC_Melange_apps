package ai.zetic.aiberry.core

/**
 * Central configuration for the Aiberry "Berry Check-in" demo.
 *
 * A guided, **conversational multimodal screener** that runs 100% on-device through
 * ZETIC Melange: the front camera + mic capture while a styled "Berry" asks open
 * questions, and live face-emotion + voice-emotion are fused into an end-of-session
 * well-being readout. Every constant here is a 1:1 port of `iOS-Aiberry`'s `AppConfig.swift`
 * so the Android and iOS apps produce identical numbers.
 */
object AppConfig {
    /** ZETIC Melange Personal Access Key (dev key for this demo).
     *  Replace with your own from https://mlange.zetic.ai -> Settings. */
    const val PERSONAL_KEY = "YOUR_MLANGE_KEY"

    /** Melange model identifiers (already hosted / uploaded). */
    object Model {
        /** 7-class wav2vec2 speech-emotion, ExecuTorch FP32, version 2. */
        const val EMOTION = "realtonypark/Wav2Vec2-Base_Emotion-Recognition"
        const val EMOTION_VERSION = 2

        /** Elena Ryumina facial-expression model (ResNet/AffectNet), version 1. */
        const val FACE = "ElenaRyumina/FaceEmotionRecognition"
        const val FACE_VERSION = 1

        // Transcript ASR is handled by ML Kit GenAI Speech Recognition (on-device,
        // platform recognizer) — see [ai.zetic.aiberry.asr.SpeechTranscriber] — so no
        // Melange Whisper model is needed here.
    }

    /** Audio capture settings. */
    const val SAMPLE_RATE = 16_000
    const val CLIP_SECONDS = 3.0
    val clipSamples: Int get() = (SAMPLE_RATE * CLIP_SECONDS).toInt() // 48_000

    /**
     * Canonical 7-emotion order shared by BOTH modalities and the fusion engine.
     * The wav2vec2 model already emits logits in this order.
     */
    val emotionLabels = listOf("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")

    /** Face / facial-expression pipeline. */
    object Face {
        const val INPUT_SIZE = 224
        // Caffe/VGGFace2-style BGR channel means (no /255, no std) — Elena Ryumina's contract.
        const val MEAN_B = 91.4953f
        const val MEAN_G = 103.8827f
        const val MEAN_R = 131.0912f
        const val INFERENCE_HZ = 3.0
        val frameIntervalMs: Long get() = (1000.0 / INFERENCE_HZ).toLong() // ~333 ms
        const val CROP_MARGIN = 0.30f
        const val TARGET_FRAMES = 30 // ~10 s at 3 Hz; saturates confidence & face weight

        // The hosted model's native logit order; remapped to [emotionLabels] before fusion.
        val nativeLabels = listOf("Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger")
        val labelToCanonical = mapOf(
            "Neutral" to "Neutral", "Happiness" to "Happy", "Sadness" to "Sad",
            "Surprise" to "Surprise", "Fear" to "Fear", "Disgust" to "Disgust", "Anger" to "Angry",
        )
    }

    /** The guided check-in script (3 open questions, asked in order). */
    object CheckIn {
        val questions = listOf(
            "How have you been feeling lately?",
            "What's been on your mind this week?",
            "Tell me about something that lifted or weighed on you recently.",
        )
        const val MIN_SECONDS = 5.0  // "Next" enabled after this
        const val MAX_SECONDS = 40.0 // auto-advance here
    }

    /**
     * Russell circumplex coordinates for each emotion (−1..+1), used to project the
     * fused distribution onto Mood (valence) and Energy (arousal).
     */
    object Affect {
        val valence = mapOf(
            "Angry" to -0.7f, "Disgust" to -0.6f, "Fear" to -0.6f, "Happy" to 0.9f,
            "Neutral" to 0.0f, "Sad" to -0.8f, "Surprise" to 0.2f,
        )
        val arousal = mapOf(
            "Angry" to 0.8f, "Disgust" to 0.3f, "Fear" to 0.8f, "Happy" to 0.6f,
            "Neutral" to 0.0f, "Sad" to -0.5f, "Surprise" to 0.8f,
        )
    }

    /** Late-fusion weights & ideals. */
    object Fusion {
        const val FACE_WEIGHT_MIN = 0.2f
        const val FACE_WEIGHT_MAX = 0.8f
        const val MOOD_WEIGHT = 0.55f
        const val ENERGY_WEIGHT = 0.30f
        const val RATE_WEIGHT = 0.15f
        const val ENERGY_IDEAL = 55f
        const val RATE_IDEAL = 60f
        const val TYPICAL_VOICED_FRACTION = 0.6f
    }
}

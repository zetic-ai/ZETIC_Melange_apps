package ai.zetic.voicevitals.core

/**
 * Central configuration for the demo.
 *
 * Every model below runs **fully on-device** through ZETIC Melange — the
 * microphone audio never leaves the phone. Swap any [name] for a client's own
 * Melange model and the rest of the app keeps working unchanged: that is the
 * whole pitch.
 */
object AppConfig {
    /** ZETIC Melange Personal Access Key (dev key supplied by ZETIC for this demo).
     *  Replace with your own from https://mlange.zetic.ai -> Settings. */
    const val PERSONAL_KEY = "YOUR_MLANGE_KEY"

    /** Melange model identifiers (already hosted / uploaded). */
    object Model {
        /** 7-class wav2vec2 SER, uploaded as **version 2** of this repo. */
        const val EMOTION = "realtonypark/Wav2Vec2-Base_Emotion-Recognition"
        const val EMOTION_VERSION = 2
        const val YAMNET = "google/Sound Classification(YAMNET)"
        const val YAMNET_VERSION = 1
    }

    /** Audio capture settings shared by every tab. */
    const val SAMPLE_RATE = 16_000
    const val CLIP_SECONDS = 3.0
    val clipSamples: Int get() = (SAMPLE_RATE * CLIP_SECONDS).toInt() // 48_000
}

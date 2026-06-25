package ai.zetic.aiberry.emotion

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ai.zetic.aiberry.core.AppConfig
import ai.zetic.aiberry.core.AudioUtils
import ai.zetic.aiberry.core.MelangeKit
import ai.zetic.aiberry.core.MelangeRuntime
import ai.zetic.aiberry.core.ModelStatus
import ai.zetic.aiberry.core.measureMs
import com.zeticai.mlange.core.model.ZeticMLangeModel

data class EmotionScore(val label: String, val probability: Float)

/**
 * Speech-emotion recognition on-device via `realtonypark/Wav2Vec2-Base_Emotion-Recognition`
 * (wav2vec2-base, Apache-2.0). Input: raw 16 kHz waveform `(1, 48000)`; output: 7 logits.
 *
 * The cleanest stand-in for a mental-health vocal biomarker: the same pipeline a
 * client would use, just with their model.
 */
class EmotionModel(private val context: Context) {
    var status by mutableStateOf<ModelStatus>(ModelStatus.Idle)
        private set
    var scores by mutableStateOf<List<EmotionScore>>(emptyList())
        private set
    var latencyMs by mutableStateOf<Double?>(null)
        private set

    private var model: ZeticMLangeModel? = null
    // Shared across all models so ZeticMLange init/run never overlap (see MelangeRuntime).
    private val queue = MelangeRuntime.executor
    private val main = Handler(Looper.getMainLooper())

    // Label order matches the model's config.id2label:
    // [angry, disgust, fear, happy, neutral, sad, surprise].
    private val labels = listOf("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")

    val top: EmotionScore? get() = scores.maxByOrNull { it.probability }

    /** Download + compile the model ahead of time (called at launch). */
    fun preload() {
        if (model != null || status.isBusy) return
        status = ModelStatus.Downloading(0f)
        queue.execute {
            try {
                ensureModel()
                main.post { status = ModelStatus.Idle }
            } catch (e: Throwable) {
                main.post { status = ModelStatus.Failed(MelangeKit.friendly(e)) }
            }
        }
    }

    /**
     * Analyze a waveform. [onResult], if given, receives the 7 softmax probabilities in
     * canonical [emotionLabels] order on the main thread — this is what the multimodal
     * fusion engine consumes (matches iOS `voice.analyze(concat) { probs in … }`).
     */
    fun analyze(samples: FloatArray, onResult: ((FloatArray) -> Unit)? = null) {
        if (status.isBusy) {
            onResult?.let { cb -> main.post { cb(FloatArray(0)) } }
            return
        }
        status = ModelStatus.Running

        queue.execute {
            try {
                val model = ensureModel()
                val n = AppConfig.clipSamples

                // 1) Trim dead air so the model's mean-pool sees mostly speech.
                val speech = AudioUtils.trimSilence(samples)
                // 2) Cover the whole utterance: average logits over overlapping windows
                //    (instead of cropping to the middle 3 s). Each window is tiled to 48000.
                val windows = AudioUtils.windows(speech, n).map { MelangeKit.fitTiling(it, n) }

                val summed = FloatArray(labels.size)
                var totalMs = 0.0
                for (window in windows) {
                    val input = MelangeKit.floatTensor(window, intArrayOf(1, n)) // (1, 48000)
                    val (outputs, ms) = measureMs { model.run(arrayOf(input)) }
                    totalMs += ms
                    val logits = outputs.firstOrNull()?.let { MelangeKit.floats(it) }
                    if (logits == null || logits.size < labels.size) {
                        throw IllegalStateException("Unexpected model output")
                    }
                    for (i in labels.indices) summed[i] += logits[i]
                }
                val logits = FloatArray(labels.size) { summed[it] / windows.size }

                // Temperature scaling (T>1) softens the model's over-confident, near
                // one-hot logits so secondary emotions are visible. Same winner; display only.
                val temperature = 2.0f
                val scaled = FloatArray(labels.size) { logits[it] / temperature }
                val probs = MelangeKit.softmax(scaled)
                val ranked = labels.indices
                    .map { EmotionScore(labels[it], probs[it]) }
                    .sortedByDescending { it.probability }

                main.post {
                    scores = ranked
                    latencyMs = totalMs
                    status = ModelStatus.Ready
                    onResult?.invoke(probs)
                }
            } catch (e: Throwable) {
                main.post {
                    status = ModelStatus.Failed(MelangeKit.friendly(e))
                    onResult?.invoke(FloatArray(0))
                }
            }
        }
    }

    @Synchronized
    private fun ensureModel(): ZeticMLangeModel {
        model?.let { return it }
        val loaded = MelangeKit.load(
            context,
            AppConfig.Model.EMOTION,
            AppConfig.Model.EMOTION_VERSION,
        ) { progress -> main.post { status = ModelStatus.Downloading(progress) } }
        model = loaded
        return loaded
    }
}

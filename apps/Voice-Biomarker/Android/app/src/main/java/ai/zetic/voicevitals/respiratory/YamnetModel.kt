package ai.zetic.voicevitals.respiratory

import android.content.Context
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ai.zetic.voicevitals.core.AppConfig
import ai.zetic.voicevitals.core.MelangeKit
import ai.zetic.voicevitals.core.MelangeRuntime
import ai.zetic.voicevitals.core.ModelStatus
import ai.zetic.voicevitals.core.measureMs
import com.zeticai.mlange.core.model.ZeticMLangeModel

data class AudioEvent(val index: Int, val name: String, val score: Float)

/**
 * Acoustic event detection on-device via `google/Sound Classification(YAMNET)`
 * (AudioSet, 521 classes, Apache-2.0). We surface the respiratory-relevant classes
 * (cough, breathing, wheeze, ...).
 */
class YamnetModel(private val context: Context) {
    var status by mutableStateOf<ModelStatus>(ModelStatus.Idle)
        private set
    var topEvents by mutableStateOf<List<AudioEvent>>(emptyList())          // overall top detections
        private set
    var respiratoryEvents by mutableStateOf<List<AudioEvent>>(emptyList())  // filtered + sorted
        private set
    var latencyMs by mutableStateOf<Double?>(null)
        private set

    private var model: ZeticMLangeModel? = null
    // Shared across all models so ZeticMLange init/run never overlap (see MelangeRuntime).
    private val queue = MelangeRuntime.executor
    private val main = Handler(Looper.getMainLooper())
    private val classNames: List<String> by lazy { loadLabels(context) }

    /** AudioSet indices for respiratory / breath-related sounds. */
    private val respiratoryIndices = intArrayOf(42, 36, 37, 44, 43, 45, 39, 41, 38, 23, 54)
    // 42 Cough, 36 Breathing, 37 Wheeze, 44 Sneeze, 43 Throat clearing,
    // 45 Sniff, 39 Gasp, 41 Snort, 38 Snoring, 23 Sigh, 54 Hiccup

    val topRespiratory: AudioEvent? get() = respiratoryEvents.firstOrNull()

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

    fun analyze(samples: FloatArray) {
        if (status.isBusy) return
        status = ModelStatus.Running

        queue.execute {
            try {
                val model = ensureModel()
                val fitted = MelangeKit.fit(samples, AppConfig.clipSamples) // 3 s window
                val input = MelangeKit.floatTensor(fitted, intArrayOf(fitted.size)) // raw waveform
                val (outputs, ms) = measureMs { model.run(arrayOf(input)) }

                // YAMNet emits several tensors (scores [N,521], embeddings [N,1024],
                // mel [M,64]). 521 is prime, so only the scores tensor's element count
                // is divisible by 521 — pick by that, not by shape metadata (unreliable)
                // or size (embeddings are larger and would be picked by mistake -> 0%).
                val scoresTensor = outputs.firstOrNull { MelangeKit.floatCount(it) % 521 == 0 }
                    ?: outputs.firstOrNull()
                    ?: throw IllegalStateException("No YAMNet score tensor in outputs")

                val flat = MelangeKit.floats(scoresTensor)
                val mean = meanOverFrames(flat, 521)

                fun named(idx: Int): String =
                    if (idx < classNames.size) classNames[idx] else "Class $idx"

                val top = mean.indices
                    .sortedByDescending { mean[it] }
                    .take(5)
                    .map { AudioEvent(it, named(it), mean[it]) }

                val resp = respiratoryIndices
                    .filter { it < mean.size }
                    .map { AudioEvent(it, named(it), mean[it]) }
                    .sortedByDescending { it.score }

                main.post {
                    topEvents = top
                    respiratoryEvents = resp
                    latencyMs = ms
                    status = ModelStatus.Ready
                }
            } catch (e: Throwable) {
                main.post { status = ModelStatus.Failed(MelangeKit.friendly(e)) }
            }
        }
    }

    @Synchronized
    private fun ensureModel(): ZeticMLangeModel {
        model?.let { return it }
        val loaded = MelangeKit.load(
            context,
            AppConfig.Model.YAMNET,
            AppConfig.Model.YAMNET_VERSION,
        ) { progress -> main.post { status = ModelStatus.Downloading(progress) } }
        model = loaded
        return loaded
    }

    companion object {
        /** YAMNet returns [frames, 521]; average the per-frame scores. */
        private fun meanOverFrames(flat: FloatArray, classes: Int): FloatArray {
            if (flat.size < classes) return flat
            val frames = flat.size / classes
            if (frames <= 1) return flat.copyOfRange(0, classes)
            val acc = FloatArray(classes)
            for (f in 0 until frames) {
                val base = f * classes
                for (c in 0 until classes) acc[c] += flat[base + c]
            }
            val inv = 1f / frames
            return FloatArray(classes) { acc[it] * inv }
        }

        /** Loads display names from `assets/yamnet_class_map.csv` (index,mid,display_name). */
        private fun loadLabels(context: Context): List<String> = try {
            context.assets.open("yamnet_class_map.csv").bufferedReader().useLines { lines ->
                lines.drop(1).mapNotNull { line ->
                    val cols = line.split(",", limit = 3)
                    if (cols.size == 3) cols[2].trim().trim('"', ' ') else null
                }.toList()
            }
        } catch (_: Throwable) {
            emptyList()
        }
    }
}

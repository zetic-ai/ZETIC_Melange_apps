package ai.zetic.aiberry.face

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ai.zetic.aiberry.core.AppConfig
import ai.zetic.aiberry.core.MelangeKit
import ai.zetic.aiberry.core.MelangeRuntime
import ai.zetic.aiberry.core.ModelStatus
import ai.zetic.aiberry.core.measureMs
import ai.zetic.aiberry.emotion.EmotionScore
import com.zeticai.mlange.core.model.ZeticMLangeModel
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.max

/**
 * Live facial-expression recognition via `ElenaRyumina/FaceEmotionRecognition` (v1).
 *
 * Per frame (throttled to ~3 Hz by [ai.zetic.aiberry.face.CameraController]): ML Kit detects +
 * crops the face -> [FacePixelTensor] builds the BGR/mean-subtracted `[1,3,224,224]` tensor ->
 * the model emits 7 logits in its native order -> softmax -> remapped to canonical order ->
 * folded into a confidence-weighted running mean. 1:1 with iOS-Aiberry's `FaceEmotionModel`.
 *
 * Detection/crop run on a private single thread; the Melange `run` is funnelled onto the
 * shared [MelangeRuntime.executor] (the SDK forbids concurrent model init/run).
 */
class FaceEmotionModel(private val context: Context) {
    var status by mutableStateOf<ModelStatus>(ModelStatus.Idle)
        private set
    var liveScores by mutableStateOf<List<EmotionScore>>(emptyList())
        private set
    var framesWithFace by mutableIntStateOf(0)
        private set
    var latencyMs by mutableStateOf<Double?>(null)
        private set

    private var model: ZeticMLangeModel? = null
    private val melange = MelangeRuntime.executor          // shared; model load + run only
    private val faceWork = Executors.newSingleThreadExecutor { r ->
        Thread(r, "aiberry-face").apply { isDaemon = true }
    }
    private val main = Handler(Looper.getMainLooper())
    private val detector = FaceDetector()
    private val busy = AtomicBoolean(false)

    private val labels = AppConfig.emotionLabels
    // native logit index -> canonical [emotionLabels] index
    private val nativeToCanonical = IntArray(7) { i ->
        labels.indexOf(AppConfig.Face.labelToCanonical[AppConfig.Face.nativeLabels[i]])
    }

    // Running sum (canonical order) of softmax*weight, and the good-frame count.
    private val summed = FloatArray(7)
    private var frames = 0

    val liveTop: EmotionScore? get() = liveScores.maxByOrNull { it.probability }

    fun preload() {
        if (model != null || status.isBusy) return
        status = ModelStatus.Downloading(0f)
        melange.execute {
            try {
                ensureModel()
                main.post { if (status !is ModelStatus.Failed) status = ModelStatus.Idle }
            } catch (e: Throwable) {
                main.post { status = ModelStatus.Failed(MelangeKit.friendly(e)) }
            }
        }
    }

    /** Clear the running accumulator at the start of a new check-in. */
    fun reset() {
        faceWork.execute {
            summed.fill(0f)
            frames = 0
            main.post {
                liveScores = emptyList()
                framesWithFace = 0
            }
        }
    }

    /**
     * Feed one camera frame. [bitmap] is the latest analysis frame (RGBA), [rotationDegrees]
     * its orientation. Dropped immediately if a previous frame is still being processed
     * (single-in-flight gate), so the live stream never piles up.
     */
    fun ingest(bitmap: Bitmap, rotationDegrees: Int) {
        if (model == null) return
        if (!busy.compareAndSet(false, true)) return
        faceWork.execute {
            try {
                val m = model ?: return@execute
                val upright = rotateUpright(bitmap, rotationDegrees)
                val face = detector.detect(upright) ?: return@execute
                val values = FacePixelTensor.bgrMeanSubtracted(face.bitmap)
                val input = MelangeKit.floatTensor(values, intArrayOf(1, 3, 224, 224))
                val (outputs, ms) = measureMs {
                    melange.submit(Callable { m.run(arrayOf(input)) }).get()
                }
                val logits = outputs.firstOrNull()?.let { MelangeKit.floats(it) } ?: return@execute
                if (logits.size < 7) return@execute
                val probs = MelangeKit.softmax(logits.copyOf(7))

                val w = max(0.2f, face.confidence)
                for (i in 0 until 7) summed[nativeToCanonical[i]] += probs[i] * w
                frames += 1

                val mean = FloatArray(7) { summed[it] / frames }
                val ranked = labels.indices
                    .map { EmotionScore(labels[it], mean[it]) }
                    .sortedByDescending { it.probability }
                val f = frames
                main.post {
                    liveScores = ranked
                    framesWithFace = f
                    latencyMs = ms
                }
            } catch (_: Throwable) {
                // Drop this frame; the next one will try again.
            } finally {
                busy.set(false)
            }
        }
    }

    /**
     * End-of-session distribution: the running sum renormalized to a probability vector in
     * canonical order, plus the good-frame count. Delivered on the main thread.
     */
    fun finalize(onResult: (distribution: FloatArray, framesWithFace: Int) -> Unit) {
        faceWork.execute {
            val f = frames
            if (f <= 0) {
                main.post { onResult(FloatArray(0), 0) }
                return@execute
            }
            val total = summed.sum()
            val dist = if (total > 0f) FloatArray(7) { summed[it] / total } else summed.copyOf()
            main.post { onResult(dist, f) }
        }
    }

    @Synchronized
    private fun ensureModel(): ZeticMLangeModel {
        model?.let { return it }
        val loaded = MelangeKit.load(
            context,
            AppConfig.Model.FACE,
            AppConfig.Model.FACE_VERSION,
        ) { progress -> main.post { status = ModelStatus.Downloading(progress) } }
        model = loaded
        return loaded
    }

    private fun rotateUpright(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
        if (rotationDegrees % 360 == 0) return bitmap
        val m = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }
}

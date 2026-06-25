package ai.zetic.demo.cameravitals.pipeline

import android.graphics.Bitmap
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.util.Size
import ai.zetic.demo.cameravitals.AppConfig
import ai.zetic.demo.cameravitals.signal.HeartRateEstimator
import ai.zetic.demo.cameravitals.util.MedianEMA
import ai.zetic.demo.cameravitals.util.MemoryProbe
import ai.zetic.demo.cameravitals.vision.FaceRoiTracker
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

class FaceState(
    val faceFound: Boolean,
    val faceBox: Rect?,
    val bufferSize: Size,
    val lowLight: Boolean,
    val framesFilled: Int,
    val capacity: Int
)

class VitalsUpdate(
    val bpm: Double?,
    val quality: Double,
    val waveform: FloatArray,
    val latencyMs: Double,
    val warmupProgress: Double
)

interface RppgPipelineListener {
    fun onFaceUpdate(state: FaceState)
    fun onVitalsUpdate(update: VitalsUpdate)
}

/**
 * Orchestrates the temporal pipeline: face crop → ring buffer → sliding-window inference → HR.
 * `process()` runs on the camera analyzer thread; inference runs on its own thread with a
 * single-flight `busy` gate (drop windows under back-pressure, never queue them).
 * Mirrors the iOS RPPGPipeline exactly.
 */
class RppgPipeline(private val model: RppgModel) {
    var listener: RppgPipelineListener? = null

    private val tracker = FaceRoiTracker()
    private val ring = FrameRingBuffer(AppConfig.FRAMES_IN, AppConfig.frameFloatCount)
    private val inferenceExecutor = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())

    private var frameCounter = 0
    private var detectCounter = 0
    private var lastBox: Rect? = null
    private val busy = AtomicBoolean(false)

    // Stitched samples + display waveform + BPM smoothing — touched only on the inference thread.
    private var rawWaveform = FloatArray(0)
    private var displayWaveform = FloatArray(0)
    private val bpmFilter = MedianEMA(5, 0.3)
    private var inferenceCount = 0

    fun reset() {
        ring.reset()
        tracker.reset()
        frameCounter = 0
        detectCounter = 0
        lastBox = null
        busy.set(false)
        inferenceExecutor.execute {
            rawWaveform = FloatArray(0)
            displayWaveform = FloatArray(0)
            bpmFilter.reset()
        }
    }

    /** Called on the camera analyzer thread for every frame. Recycles `bitmap` before returning. */
    fun process(bitmap: Bitmap) {
        try {
            val bufferSize = Size(bitmap.width, bitmap.height)

            // Detect + publish face state every 3rd frame (~10 Hz); crop EVERY frame for 30 fps.
            detectCounter++
            val doDetect = detectCounter % 3 == 0
            if (doDetect) lastBox = tracker.detect(bitmap)

            val box = lastBox
            if (box == null) {
                // Face lost beyond hysteresis: drop the partial window so we never stitch a gap.
                ring.reset()
                frameCounter = 0
                if (doDetect) {
                    emitFace(FaceState(false, null, bufferSize, false, ring.filled, AppConfig.FRAMES_IN))
                }
                return
            }

            val cropped = FrameCropper.crop(bitmap, box) ?: return
            val lowLight = cropped.meanLuma < AppConfig.LOW_LUMA_THRESHOLD
            ring.append(cropped.planarRGB)
            frameCounter++

            if (doDetect) {
                emitFace(FaceState(true, box, bufferSize, lowLight, ring.filled, AppConfig.FRAMES_IN))
            }

            if (ring.isFull && frameCounter >= AppConfig.STRIDE && busy.compareAndSet(false, true)) {
                frameCounter = 0
                val snapshot = ring.snapshot()
                if (snapshot == null) {
                    busy.set(false)
                } else {
                    inferenceExecutor.execute { runInference(snapshot) }
                }
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun runInference(snapshot: FloatArray) {
        try {
            val tensor = TensorBuilder.build(snapshot) ?: return
            val chunkOut = try {
                model.infer(tensor)
            } catch (e: Exception) {
                return   // transient failure: skip this window
            }

            inferenceCount++
            if (inferenceCount % 10 == 1) MemoryProbe.log("infer #$inferenceCount")

            // Stitch this chunk's samples into the rolling analysis buffer.
            rawWaveform += chunkOut
            if (rawWaveform.size > AppConfig.ANALYSIS_SAMPLES) {
                rawWaveform = rawWaveform.copyOfRange(rawWaveform.size - AppConfig.ANALYSIS_SAMPLES, rawWaveform.size)
            }

            val progress = min(rawWaveform.size.toDouble() / AppConfig.MIN_ANALYSIS_SAMPLES, 1.0)

            var shownBPM = bpmFilter.value
            var quality = 0.0
            if (rawWaveform.size >= 60) {
                val result = HeartRateEstimator.estimate(rawWaveform, AppConfig.FPS)
                if (result != null) {
                    displayWaveform = result.filtered.takeLast(AppConfig.ANALYSIS_SAMPLES).toFloatArray()
                    quality = result.quality
                    // Update the shown HR on ANY physiological peak once warmed up — the
                    // median+EMA filter handles noise, quality drives the badge separately.
                    if (rawWaveform.size >= AppConfig.MIN_ANALYSIS_SAMPLES &&
                        result.bpm >= AppConfig.MIN_BPM && result.bpm <= AppConfig.MAX_BPM
                    ) {
                        shownBPM = bpmFilter.update(result.bpm)
                    }
                }
            }

            val update = VitalsUpdate(shownBPM, quality, displayWaveform, model.lastLatencyMs, progress)
            emitVitals(update)
        } finally {
            busy.set(false)
        }
    }

    private fun emitFace(state: FaceState) {
        main.post { listener?.onFaceUpdate(state) }
    }

    private fun emitVitals(update: VitalsUpdate) {
        main.post { listener?.onVitalsUpdate(update) }
    }
}

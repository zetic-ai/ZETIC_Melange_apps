package ai.zetic.demo.cameravitals.state

import android.app.Application
import android.graphics.Rect
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Size
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.zetic.demo.cameravitals.AppConfig
import ai.zetic.demo.cameravitals.camera.CameraController
import ai.zetic.demo.cameravitals.pipeline.FaceState
import ai.zetic.demo.cameravitals.pipeline.RppgModel
import ai.zetic.demo.cameravitals.pipeline.RppgPipeline
import ai.zetic.demo.cameravitals.pipeline.RppgPipelineListener
import ai.zetic.demo.cameravitals.pipeline.VitalsUpdate
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

/**
 * Bridges the pipeline to Compose. State is exposed via `mutableStateOf`; all mutations happen on
 * the main thread (the pipeline marshals its callbacks to main; model-load callbacks are reposted).
 * Mirrors the iOS VitalsViewModel.
 */
class VitalsViewModel(app: Application) : AndroidViewModel(app), RppgPipelineListener {
    var state by mutableStateOf<MeasurementState>(MeasurementState.LoadingModel(0f)); private set
    var bpm by mutableStateOf<Double?>(null); private set
    var quality by mutableStateOf(0.0); private set
    var waveform by mutableStateOf(FloatArray(0)); private set
    var latencyMs by mutableStateOf(0.0); private set

    var faceFound by mutableStateOf(false); private set
    var faceBox by mutableStateOf<Rect?>(null); private set
    var bufferSize by mutableStateOf(Size(0, 0)); private set
    var lowLight by mutableStateOf(false); private set
    var warmupProgress by mutableStateOf(0.0); private set

    var isMeasuring by mutableStateOf(false); private set
    var measureProgress by mutableStateOf(0.0); private set
    var report by mutableStateOf<MeasurementReport?>(null); private set

    val camera = CameraController(app.applicationContext)
    private val model = RppgModel()
    private var pipeline: RppgPipeline? = null
    private val main = Handler(Looper.getMainLooper())

    private val measureSamples = mutableListOf<Pair<Double, Double>>()  // (bpm, quality)
    private var measureStartMs = 0L
    private var measureJob: Job? = null

    val isReady: Boolean
        get() = state !is MeasurementState.LoadingModel &&
            state !is MeasurementState.PermissionDenied &&
            state !is MeasurementState.ErrorState

    // MARK: - Lifecycle

    fun onPermissionResult(granted: Boolean) {
        if (!granted) {
            state = MeasurementState.PermissionDenied
            return
        }
        if (pipeline == null) loadModel()
    }

    private fun loadModel() {
        state = MeasurementState.LoadingModel(0f)
        model.load(
            getApplication<Application>().applicationContext,
            onProgress = { p ->
                main.post {
                    if (state is MeasurementState.LoadingModel) state = MeasurementState.LoadingModel(p)
                }
            },
            onResult = { err ->
                main.post {
                    if (err != null) {
                        state = MeasurementState.ErrorState(err.message ?: "Model load failed")
                    } else {
                        val pipe = RppgPipeline(model)
                        pipe.listener = this
                        pipeline = pipe
                        camera.onFrame = { bmp -> pipe.process(bmp) }
                        state = MeasurementState.Warmup(0)
                    }
                }
            }
        )
    }

    fun retry() {
        if (state is MeasurementState.ErrorState) loadModel()
    }

    fun stop() {
        camera.stop()
        cancelMeasurement()
    }

    override fun onCleared() {
        super.onCleared()
        camera.shutdown()
    }

    // MARK: - Guided measurement

    fun startMeasurement() {
        if (!isReady || isMeasuring) return
        measureSamples.clear()
        measureStartMs = SystemClock.elapsedRealtime()
        measureProgress = 0.0
        report = null
        isMeasuring = true
        measureJob = viewModelScope.launch {
            while (isMeasuring) {
                val elapsed = (SystemClock.elapsedRealtime() - measureStartMs) / 1000.0
                measureProgress = (elapsed / AppConfig.MEASURE_DURATION_SEC).coerceAtMost(1.0)
                if (elapsed >= AppConfig.MEASURE_DURATION_SEC) {
                    finishMeasurement()
                    break
                }
                delay(100)
            }
        }
    }

    fun cancelMeasurement() {
        measureJob?.cancel()
        measureJob = null
        isMeasuring = false
        measureProgress = 0.0
    }

    private fun finishMeasurement() {
        isMeasuring = false
        measureProgress = 1.0

        val good = measureSamples.filter { it.second > AppConfig.DISPLAY_QUALITY_FLOOR }
        val used = good.ifEmpty { measureSamples }
        if (used.isEmpty()) {
            report = null
            return
        }
        val bpms = used.map { it.first }
        val avg = bpms.average()
        report = MeasurementReport(
            avgBPM = avg.roundToInt(),
            minBPM = (bpms.minOrNull() ?: avg).roundToInt(),
            maxBPM = (bpms.maxOrNull() ?: avg).roundToInt(),
            avgQuality = used.map { it.second }.average(),
            series = bpms
        )
    }

    fun dismissReport() {
        report = null
    }

    // MARK: - RppgPipelineListener (called on the main thread)

    override fun onFaceUpdate(state: FaceState) {
        faceFound = state.faceFound
        faceBox = state.faceBox
        bufferSize = state.bufferSize
        lowLight = state.lowLight
    }

    override fun onVitalsUpdate(update: VitalsUpdate) {
        bpm = update.bpm
        quality = update.quality
        waveform = update.waveform
        latencyMs = update.latencyMs
        warmupProgress = update.warmupProgress

        if (state is MeasurementState.Warmup && update.warmupProgress >= 1.0) {
            state = MeasurementState.Live
        }

        if (isMeasuring) {
            update.bpm?.let { measureSamples.add(it to update.quality) }
        }
    }
}

package ai.zetic.demo.posemotion.state

import ai.zetic.demo.posemotion.AppConfig
import ai.zetic.demo.posemotion.pipeline.BenchmarkSnapshot
import ai.zetic.demo.posemotion.pipeline.FrameResult
import ai.zetic.demo.posemotion.pipeline.MelangeModel
import ai.zetic.demo.posemotion.pipeline.PosePipeline
import ai.zetic.demo.posemotion.video.ClipFrameSource
import ai.zetic.demo.posemotion.video.ClipLocator
import ai.zetic.demo.posemotion.video.VideoFrameDecoder
import android.app.Application
import android.graphics.PointF
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import androidx.lifecycle.AndroidViewModel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

data class ModelLoadState(
    val label: String,
    val progress: Float = 0f,
    val loaded: Boolean = false,
    val error: String? = null,
    val optional: Boolean = false,
)

class DemoViewModel(app: Application) : AndroidViewModel(app) {
    enum class Phase { LOADING, MISSING_CLIP, RUNNING }

    private val inferenceThread = HandlerThread("inference").apply { start() }
    private val handler = Handler(inferenceThread.looper)
    private val mainHandler = Handler(Looper.getMainLooper())
    private val pipeline = PosePipeline()
    private var source: ClipFrameSource? = null

    private val _phase = MutableStateFlow(Phase.LOADING)
    val phase = _phase.asStateFlow()

    private val _loadStates = MutableStateFlow(
        listOf(
            ModelLoadState("YOLO26n · detector"),
            ModelLoadState("RTMPose-s · 2D pose"),
            ModelLoadState("MotionBERT-Lite · 3D lift", optional = true),
        )
    )
    val loadStates = _loadStates.asStateFlow()

    private val _mode = MutableStateFlow(ClipFrameSource.Mode.BENCHMARK)
    val mode = _mode.asStateFlow()

    private val _show3D = MutableStateFlow(true)
    val show3D = _show3D.asStateFlow()

    private val _frame = MutableStateFlow<FrameResult?>(null)
    val frame = _frame.asStateFlow()

    private val _stats = MutableStateFlow(BenchmarkSnapshot())
    val stats = _stats.asStateFlow()

    private val _ballTrail = MutableStateFlow<List<PointF>>(emptyList())
    val ballTrail = _ballTrail.asStateFlow()

    private val _availableClips = MutableStateFlow<List<String>>(emptyList())
    val availableClips = _availableClips.asStateFlow()

    private val _selectedClip = MutableStateFlow(AppConfig.CLIP_FILES.first())
    val selectedClip = _selectedClip.asStateFlow()

    val liftAvailable: Boolean get() = pipeline.lift.isLoaded

    private data class Spec(val model: MelangeModel, val name: String, val version: Int)
    private val specs = listOf(
        Spec(pipeline.detector, AppConfig.DETECTOR_NAME, AppConfig.DETECTOR_VERSION),
        Spec(pipeline.pose, AppConfig.POSE_NAME, AppConfig.POSE_VERSION),
        Spec(pipeline.lift, AppConfig.LIFT_NAME, AppConfig.LIFT_VERSION),
    )

    init {
        pipeline.onResult = { result, snapshot ->
            mainHandler.post { apply(result, snapshot) }
        }
        // Sequential loads on the inference thread: SDK native init is thread-affine,
        // so models must be constructed on the thread that runs them.
        handler.post {
            specs.forEachIndexed { index, spec -> loadOne(index, spec) }
            pipeline.warmup()
            mainHandler.post { startIfReady() }
        }
    }

    fun retry(index: Int) {
        updateRow(index) { it.copy(error = null, progress = 0f) }
        handler.post {
            loadOne(index, specs[index])
            pipeline.warmup()
            mainHandler.post { startIfReady() }
        }
    }

    fun toggle3D() {
        _show3D.value = !_show3D.value
    }

    fun setMode(newMode: ClipFrameSource.Mode) {
        if (newMode == _mode.value) return
        _mode.value = newMode
        if (_phase.value != Phase.RUNNING) return
        source?.stop()
        handler.post { pipeline.resetSession() }
        _ballTrail.value = emptyList()
        startSource()
    }

    fun setClip(fileName: String) {
        if (fileName == _selectedClip.value) return
        _selectedClip.value = fileName
        if (_phase.value != Phase.RUNNING) return
        source?.stop()
        handler.post { pipeline.resetSession() }
        _ballTrail.value = emptyList()
        _frame.value = null
        startSource()
    }

    /** Re-checks for clips (missing-clip screen "check again"). */
    fun recheckClip() {
        if (_phase.value != Phase.MISSING_CLIP) return
        val clips = ClipLocator.availableClips(getApplication())
        if (clips.isNotEmpty()) {
            _availableClips.value = clips
            if (_selectedClip.value !in clips) _selectedClip.value = clips.first()
            _phase.value = Phase.RUNNING
            startSource()
        }
    }

    override fun onCleared() {
        source?.stop()
        handler.post { pipeline.deinitAll() }
        inferenceThread.quitSafely()
    }

    // MARK: - Internals

    /** Runs on the inference thread. */
    private fun loadOne(index: Int, spec: Spec) {
        try {
            spec.model.load(getApplication(), spec.name, spec.version) { progress ->
                mainHandler.post { updateRow(index) { it.copy(progress = progress) } }
            }
            mainHandler.post { updateRow(index) { it.copy(loaded = true, error = null) } }
        } catch (t: Throwable) {
            mainHandler.post {
                updateRow(index) { it.copy(error = t.message ?: t.javaClass.simpleName) }
            }
        }
    }

    private fun startIfReady() {
        if (_phase.value != Phase.LOADING) return
        val allRequired = _loadStates.value.all { it.loaded || (it.optional && it.error != null) }
        if (!allRequired) return
        val clips = ClipLocator.availableClips(getApplication())
        _availableClips.value = clips
        if (clips.isEmpty()) {
            _phase.value = Phase.MISSING_CLIP
            return
        }
        if (_selectedClip.value !in clips) _selectedClip.value = clips.first()
        _phase.value = Phase.RUNNING
        startSource()
    }

    private fun startSource() {
        // Fresh ClipSource per start: the decoder closes the asset fd on release.
        val clip = ClipLocator.locate(getApplication(), _selectedClip.value) ?: run {
            _phase.value = Phase.MISSING_CLIP
            return
        }
        val s = ClipFrameSource(VideoFrameDecoder(clip), _mode.value, handler)
        s.onFrame = { frame -> pipeline.process(frame) }
        source = s
        s.start()
    }

    private fun updateRow(index: Int, transform: (ModelLoadState) -> ModelLoadState) {
        _loadStates.value = _loadStates.value.mapIndexed { i, s ->
            if (i == index) transform(s) else s
        }
    }

    private fun apply(result: FrameResult, snapshot: BenchmarkSnapshot) {
        _frame.value = result
        _stats.value = snapshot
        result.ballBox?.let { ball ->
            val trail = _ballTrail.value + PointF(ball.centerX(), ball.centerY())
            _ballTrail.value = trail.takeLast(AppConfig.BALL_TRAIL_LENGTH)
        }
    }
}

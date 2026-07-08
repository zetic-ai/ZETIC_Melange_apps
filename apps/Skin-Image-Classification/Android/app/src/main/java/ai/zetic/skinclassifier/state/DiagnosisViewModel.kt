package ai.zetic.skinclassifier.state

import android.app.Application
import android.graphics.Bitmap
import android.os.Handler
import android.os.Looper
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import ai.zetic.skinclassifier.classifier.SkinClassifier
import ai.zetic.skinclassifier.core.MelangeKit
import ai.zetic.skinclassifier.core.ModelStatus
import ai.zetic.skinclassifier.model.Classification
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/** Where the analysis stands for the current photo. Mirrors iOS `AnalysisState`. */
sealed interface AnalysisState {
    data object None : AnalysisState
    data object Classifying : AnalysisState
    data object Done : AnalysisState
    data class Failed(val message: String) : AnalysisState
}

/**
 * Drives the screen flow: load the classifier (with download progress) -> capture a photo ->
 * classify on-device -> show the verdict. 1:1 with the iOS `DiagnosisViewModel`.
 *
 * Orchestration runs on a private single-thread [work] executor; the classifier internally
 * funnels SDK calls onto the shared Melange runtime thread, so [work] must be distinct from it
 * to avoid a submit-from-within-executor deadlock. UI state is posted back on the main thread.
 */
class DiagnosisViewModel(app: Application) : AndroidViewModel(app) {

    var classifierStatus by mutableStateOf<ModelStatus>(ModelStatus.Idle)
        private set
    var analysis by mutableStateOf<AnalysisState>(AnalysisState.None)
        private set
    var image by mutableStateOf<Bitmap?>(null)
        private set
    var classification by mutableStateOf<Classification?>(null)
        private set

    val canAnalyze: Boolean get() = classifierStatus.isReady

    private val classifier = SkinClassifier(app)
    private val work = Executors.newSingleThreadExecutor { r ->
        Thread(r, "skin-vm").apply { isDaemon = true }
    }
    private val main = Handler(Looper.getMainLooper())
    private val analyzeGeneration = AtomicInteger(0)
    private var bootstrapStarted = false

    /** Kick off the one-time classifier load. Safe to call repeatedly. */
    fun bootstrap() {
        if (bootstrapStarted) return
        bootstrapStarted = true
        classifierStatus = ModelStatus.Preparing
        work.execute {
            try {
                classifier.ensureLoaded { progress ->
                    val status = if (progress > 0f && progress < 1f) {
                        ModelStatus.Downloading(progress)
                    } else {
                        ModelStatus.Preparing
                    }
                    main.post { classifierStatus = status }
                }
                main.post { classifierStatus = ModelStatus.Ready }
            } catch (t: Throwable) {
                main.post { classifierStatus = ModelStatus.Failed(MelangeKit.friendly(t)) }
            }
        }
    }

    /** Deinit the (failed) model and try the load again from scratch. */
    fun retryLoad() {
        work.execute { classifier.close() }
        bootstrapStarted = false
        classifierStatus = ModelStatus.Idle
        bootstrap()
    }

    /** Classify [bitmap], superseding any in-flight analysis. */
    fun analyze(bitmap: Bitmap) {
        val generation = analyzeGeneration.incrementAndGet()
        image = bitmap
        classification = null
        analysis = AnalysisState.Classifying
        work.execute {
            try {
                val result = classifier.classify(bitmap)
                if (generation != analyzeGeneration.get()) return@execute
                main.post {
                    if (generation != analyzeGeneration.get()) return@post
                    classification = result
                    analysis = AnalysisState.Done
                }
            } catch (t: Throwable) {
                if (generation != analyzeGeneration.get()) return@execute
                main.post {
                    if (generation != analyzeGeneration.get()) return@post
                    analysis = AnalysisState.Failed(MelangeKit.friendly(t))
                }
            }
        }
    }

    /** Re-run analysis on the current image (used by the Retry button on a failure). */
    fun retryAnalyze() {
        image?.let { analyze(it) }
    }

    /** Discard the result and return to the capture screen. */
    fun reset() {
        analyzeGeneration.incrementAndGet()
        image = null
        classification = null
        analysis = AnalysisState.None
    }

    override fun onCleared() {
        work.execute { classifier.close() }
        super.onCleared()
    }
}

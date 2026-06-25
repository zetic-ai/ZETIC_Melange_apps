package ai.zetic.aiberry.session

import android.app.Application
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.zetic.aiberry.core.AppConfig
import ai.zetic.aiberry.core.AudioRecorder
import ai.zetic.aiberry.core.AudioUtils
import ai.zetic.aiberry.core.ModelStatus
import ai.zetic.aiberry.emotion.EmotionModel
import ai.zetic.aiberry.face.CameraController
import ai.zetic.aiberry.face.FaceEmotionModel
import ai.zetic.aiberry.asr.SpeechTranscriber
import ai.zetic.aiberry.ui.Haptics
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/** The guided check-in flow, mirroring iOS-Aiberry's `CheckInSession` phases. */
sealed interface Phase {
    data object Idle : Phase          // Landing screen
    data object Intro : Phase         // "How it works" / consent
    data class Question(val index: Int) : Phase
    data object Analyzing : Phase
    data class Insights(val report: ScreeningReport) : Phase
}

/**
 * Orchestrates the whole "Berry Check-in": owns the three on-device models (voice, face,
 * Whisper), the recorder, camera and audio session, drives the question state machine, and
 * fuses everything into a [ScreeningReport]. Direct port of `CheckInSession.swift`.
 *
 * UI observes Compose state (`phase`, `elapsed`) here and the models' own observable state
 * (`recorder.level`, `face.liveTop`, …) directly.
 */
class CheckInViewModel(app: Application) : AndroidViewModel(app) {

    val voice = EmotionModel(app)
    val face = FaceEmotionModel(app)
    val transcriber = SpeechTranscriber(app)
    val recorder = AudioRecorder()
    val camera = CameraController(app)
    private val sessionAudio = SessionAudio(app)
    private val speaker = QuestionSpeaker(app)

    var phase by mutableStateOf<Phase>(Phase.Idle)
        private set
    var elapsed by mutableDoubleStateOf(0.0)
        private set

    /** The current question revealed word-by-word as TTS speaks it; the full text
     *  once recording starts. */
    var spokenText by mutableStateOf("")
        private set
    /** True while the question is being read aloud (before recording the answer). */
    var isSpeaking by mutableStateOf(false)
        private set

    private val voiceClips = mutableListOf<FloatArray>()
    private var timerJob: Job? = null

    // ---- Derived UI state ----
    val questions: List<String> get() = AppConfig.CheckIn.questions
    val totalQuestions: Int get() = questions.size
    val questionIndex: Int get() = (phase as? Phase.Question)?.index ?: 0
    val currentQuestion: String get() = questions.getOrElse(questionIndex) { "" }
    val canAdvance: Boolean get() = elapsed >= AppConfig.CheckIn.MIN_SECONDS
    val countdownProgress: Double get() = (elapsed / AppConfig.CheckIn.MAX_SECONDS).coerceIn(0.0, 1.0)
    val isRecording: Boolean get() = recorder.isRecording
    val micLevel: Float get() = recorder.level
    val liveEmotion: String? get() = face.liveTop?.takeIf { it.probability > 0f }?.label

    val modelsReady: Boolean
        get() = !voice.status.isBusy && !face.status.isBusy &&
            !voice.status.isFailure && !face.status.isFailure

    /** A required model (face/voice) failed to load — typically a first-launch network issue. */
    val modelsFailed: Boolean
        get() = voice.status.isFailure || face.status.isFailure

    /** First load error worth surfacing (e.g. captive portal / offline on first download). */
    val loadError: String?
        get() = (voice.status as? ModelStatus.Failed)?.message
            ?: (face.status as? ModelStatus.Failed)?.message
            ?: (transcriber.status as? ModelStatus.Failed)?.message

    /**
     * Download + compile all models. Called at launch and re-callable as a Retry: each model's
     * preload() re-attempts from a Failed state, so once connectivity is restored (e.g. after
     * signing into a Wi-Fi captive portal) this recovers without reinstalling.
     */
    fun preloadAll() {
        voice.preload()
        face.preload()
        transcriber.preload()
    }

    // ---- Transitions ----
    fun showIntro() { phase = Phase.Intro }

    fun begin() {
        voiceClips.clear()
        face.reset()
        sessionAudio.begin()
        phase = Phase.Question(0)
        startQuestion()
    }

    /** User tapped "Next" / "See results" — stop the current recording early. */
    fun advance() {
        if (!canAdvance) return
        recorder.stop() // delivers captured samples -> finishQuestion(...)
    }

    fun cancel() {
        teardownCapture()
        voiceClips.clear()
        phase = Phase.Idle
    }

    fun restart() {
        teardownCapture()
        voiceClips.clear()
        phase = Phase.Idle
    }

    // ---- Per-question capture ----
    private fun startQuestion() {
        elapsed = 0.0
        isSpeaking = true
        spokenText = ""
        // Read the question aloud, revealing it word-by-word, then start recording.
        speaker.speak(
            currentQuestion,
            onWord = { prefix -> spokenText = prefix },
            onDone = { beginRecording() },
        )
    }

    /** Called once the question has finished being spoken — start capturing the answer. */
    private fun beginRecording() {
        if (phase !is Phase.Question) return // ignore if the user ended/advanced
        isSpeaking = false
        spokenText = currentQuestion
        elapsed = 0.0
        startTimer()
        recorder.record(autoStopSeconds = AppConfig.CheckIn.MAX_SECONDS) { samples ->
            finishQuestion(samples)
        }
    }

    private fun finishQuestion(samples: FloatArray) {
        stopTimer()
        voiceClips.add(samples)
        val next = questionIndex + 1
        if (next < questions.size) {
            phase = Phase.Question(next)
            startQuestion()
        } else {
            analyze()
        }
    }

    private fun analyze() {
        stopTimer()
        camera.unbind()
        sessionAudio.end()
        phase = Phase.Analyzing

        val clips = voiceClips.toList()
        val concat = flatten(clips)
        val voiced = AudioUtils.voicedFraction(concat)

        // Collect the three modalities; all callbacks land on the main thread, so a simple
        // countdown is race-free.
        var faceDist = FloatArray(0)
        var faceFrames = 0
        var voiceProbs = FloatArray(0)
        var transcripts = List(clips.size) { "" }
        var remaining = 3

        fun done() {
            remaining -= 1
            if (remaining > 0) return
            val pairs = clips.indices.map {
                QAPair(questions.getOrElse(it) { "" }, transcripts.getOrElse(it) { "" })
            }
            val report = FusionEngine.fuse(
                face = faceDist, faceFrames = faceFrames,
                voice = voiceProbs, voicedFraction = voiced,
                transcript = pairs,
            )
            Haptics.success(getApplication())
            phase = Phase.Insights(report)
        }

        face.finalize { dist, frames -> faceDist = dist; faceFrames = frames; done() }
        voice.analyze(concat) { probs -> voiceProbs = probs; done() }
        transcriber.transcribeAll(clips) { texts -> transcripts = texts; done() }
    }

    private fun startTimer() {
        stopTimer()
        timerJob = viewModelScope.launch {
            while (true) {
                delay(100)
                elapsed += 0.1
            }
        }
    }

    private fun stopTimer() {
        timerJob?.cancel()
        timerJob = null
    }

    private fun teardownCapture() {
        stopTimer()
        speaker.cancel()
        isSpeaking = false
        recorder.stop()
        camera.unbind()
        sessionAudio.end()
    }

    private fun flatten(clips: List<FloatArray>): FloatArray {
        val total = clips.sumOf { it.size }
        val out = FloatArray(total)
        var i = 0
        for (c in clips) { System.arraycopy(c, 0, out, i, c.size); i += c.size }
        return out
    }

    override fun onCleared() {
        teardownCapture()
        speaker.shutdown()
        super.onCleared()
    }
}

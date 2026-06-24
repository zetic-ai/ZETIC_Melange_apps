package ai.zetic.aiberry.asr

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.os.ParcelFileDescriptor
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ai.zetic.aiberry.core.ModelStatus
import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.audio.AudioSource
import com.google.mlkit.genai.speechrecognition.SpeechRecognition
import com.google.mlkit.genai.speechrecognition.SpeechRecognizer
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerRequest
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerResponse
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import kotlin.math.min

/**
 * On-device English transcription for the Transcript tab, using **ML Kit GenAI
 * Speech Recognition** (BASIC mode = the platform recognizer, on-device, API 31+).
 * This is the Android parallel to iOS-Aiberry's on-device `SFSpeechRecognizer`.
 *
 * The recorder already owns a clean 16 kHz / mono / Float32 PCM buffer per answer
 * (also used by the voice-emotion model). Rather than running a live recognizer that
 * would contend with the recorder for the mic (Android silences one of two capture
 * clients), we replay each recorded clip into the recognizer through an
 * [AudioSource.fromPfd] pipe, paced at real time as ML Kit requires.
 *
 * ## Safety
 * [transcribeAll] NEVER throws. On any failure (model unavailable on this
 * device/locale, download failure, recognition error) it delivers a list of "" of the
 * same length as the input clips on the MAIN thread and sets [status] to Failed.
 * Per-clip failures are isolated. The Transcript tab still shows the questions and a
 * per-answer "transcription unavailable" note in that case.
 */
class SpeechTranscriber(private val context: Context) {

    companion object {
        private const val TAG = "SpeechTranscriber"
        // Real-time pacing: ML Kit requires ~16k samples (≈32 KB) per second. We feed
        // 100 ms chunks (1600 samples) and sleep 100 ms between them.
        private const val CHUNK_SAMPLES = 1600
        private const val CHUNK_MS = 100L
    }

    var status by mutableStateOf<ModelStatus>(ModelStatus.Idle)
        private set

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val main = Handler(Looper.getMainLooper())

    private fun newRecognizer(): SpeechRecognizer {
        val options = SpeechRecognizerOptions.Builder().apply {
            locale = Locale.US
            preferredMode = SpeechRecognizerOptions.Mode.MODE_BASIC
        }.build()
        return SpeechRecognition.getClient(options)
    }

    /** Warm up: confirm the on-device model is present, downloading it if allowed. */
    fun preload() {
        if (status.isBusy) return
        status = ModelStatus.Downloading(0f)
        scope.launch {
            val recognizer = newRecognizer()
            try {
                val ok = ensureModel(recognizer)
                main.post { status = if (ok) ModelStatus.Ready else ModelStatus.Failed(unavailableMsg()) }
            } catch (e: Throwable) {
                Log.w(TAG, "Speech model preload failed", e)
                main.post { status = ModelStatus.Failed(unavailableMsg()) }
            } finally {
                runCatching { recognizer.close() }
            }
        }
    }

    /**
     * Transcribe each clip in order; deliver one string per clip on the MAIN thread.
     * MUST NOT throw. On total failure, delivers [""] * clips.size.
     */
    fun transcribeAll(clips: List<FloatArray>, onResult: (List<String>) -> Unit) {
        val empty = List(clips.size) { "" }
        if (clips.isEmpty()) {
            main.post { onResult(empty) }
            return
        }
        status = ModelStatus.Running
        scope.launch {
            val results = ArrayList<String>(clips.size)
            var hadFailure = false
            val recognizer = newRecognizer()
            try {
                if (!ensureModel(recognizer)) {
                    main.post { status = ModelStatus.Failed(unavailableMsg()); onResult(empty) }
                    return@launch
                }
                for (clip in clips) {
                    val text = try {
                        transcribeOne(recognizer, clip)
                    } catch (e: Throwable) {
                        Log.w(TAG, "Clip transcription failed", e)
                        hadFailure = true
                        ""
                    }
                    results.add(text)
                }
            } catch (e: Throwable) {
                Log.w(TAG, "Transcription failed", e)
                results.clear()
                results.addAll(empty)
                hadFailure = true
            } finally {
                runCatching { recognizer.close() }
            }
            val finalResults = if (results.size == clips.size) results else empty
            main.post {
                status = if (hadFailure && finalResults.all { it.isEmpty() }) {
                    ModelStatus.Failed(unavailableMsg())
                } else {
                    ModelStatus.Ready
                }
                onResult(finalResults)
            }
        }
    }

    /** Ensure the on-device recognition model is AVAILABLE; download if DOWNLOADABLE. */
    private suspend fun ensureModel(recognizer: SpeechRecognizer): Boolean {
        return when (recognizer.checkStatus()) {
            FeatureStatus.AVAILABLE -> true
            FeatureStatus.UNAVAILABLE -> false
            else -> { // DOWNLOADABLE / DOWNLOADING
                var ok = true
                recognizer.download().collect { s ->
                    when (s) {
                        is DownloadStatus.DownloadFailed -> ok = false
                        is DownloadStatus.DownloadProgress ->
                            main.post { status = ModelStatus.Downloading(0f) }
                        else -> { /* DownloadCompleted */ }
                    }
                }
                ok && recognizer.checkStatus() == FeatureStatus.AVAILABLE
            }
        }
    }

    /** Replay one recorded clip into the recognizer and return its final transcript. */
    private suspend fun transcribeOne(recognizer: SpeechRecognizer, clip: FloatArray): String {
        val pipe = ParcelFileDescriptor.createPipe()
        val readEnd = pipe[0]
        val writeEnd = pipe[1]

        // Feed PCM into the write end at real time on a separate coroutine.
        val writer = scope.launch { writePcm(writeEnd, clip) }

        val transcript = StringBuilder()
        try {
            val request = SpeechRecognizerRequest.Builder().apply {
                audioSource = AudioSource.fromPfd(readEnd)
            }.build()
            recognizer.startRecognition(request).collect { response ->
                when (response) {
                    is SpeechRecognizerResponse.FinalTextResponse ->
                        transcript.append(response.text)
                    else -> { /* PartialText / Completed / Error */ }
                }
            }
        } finally {
            writer.cancel()
            runCatching { readEnd.close() }
        }
        return transcript.toString().trim()
    }

    /** Write [clip] as headerless 16-bit PCM to [writeEnd], paced at real time. */
    private suspend fun writePcm(writeEnd: ParcelFileDescriptor, clip: FloatArray) {
        ParcelFileDescriptor.AutoCloseOutputStream(writeEnd).use { out ->
            val buf = ByteArray(CHUNK_SAMPLES * 2)
            var i = 0
            while (i < clip.size) {
                val n = min(CHUNK_SAMPLES, clip.size - i)
                for (j in 0 until n) {
                    val s = (clip[i + j].coerceIn(-1f, 1f) * 32767f).toInt()
                    buf[j * 2] = (s and 0xFF).toByte()
                    buf[j * 2 + 1] = ((s shr 8) and 0xFF).toByte()
                }
                out.write(buf, 0, n * 2)
                out.flush()
                i += n
                delay(CHUNK_MS)
            }
        } // closing the stream signals end-of-audio so recognition completes
    }

    private fun unavailableMsg() =
        "On-device speech recognition is unavailable on this device or locale."
}

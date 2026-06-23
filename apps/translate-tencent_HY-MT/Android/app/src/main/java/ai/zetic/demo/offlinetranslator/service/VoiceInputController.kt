package ai.zetic.demo.offlinetranslator.service

import android.util.Log
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.audio.AudioSource
import com.google.mlkit.genai.speechrecognition.SpeechRecognition
import com.google.mlkit.genai.speechrecognition.SpeechRecognizer
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.SpeechRecognizerResponse
import com.google.mlkit.genai.speechrecognition.speechRecognizerOptions
import com.google.mlkit.genai.speechrecognition.speechRecognizerRequest
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Offline speech-to-text via ML Kit GenAI Speech Recognition (BASIC mode). Captures the mic with
 * `AudioSource.fromMic()` and streams partial/final transcripts. The feature model downloads once
 * on first use (needs network that one time); afterwards it works fully offline.
 *
 * Coroutine-based (not blocking like the ZeticMLange engine), so it runs on [scope] (the
 * ViewModel's scope) — never on the reserved single-thread engine executor.
 */
class VoiceInputController(private val scope: CoroutineScope) {
    private var client: SpeechRecognizer? = null
    private var job: Job? = null

    // Set by stop() so a stop that lands during the async init / one-time model download (before
    // the recognizer is actually listening) still takes effect instead of being a no-op.
    @Volatile private var stopRequested = false
    @Volatile private var listening = false

    fun start(
        locale: Locale,
        onPartial: (String) -> Unit,
        onFinal: (String) -> Unit,
        onError: (String) -> Unit,
    ) {
        if (job != null) return
        stopRequested = false
        listening = false
        job = scope.launch {
            try {
                val options = speechRecognizerOptions {
                    this.locale = locale
                    preferredMode = SpeechRecognizerOptions.Mode.MODE_BASIC
                }
                val recognizer = SpeechRecognition.getClient(options).also { client = it }
                if (stopRequested) { onFinal(""); return@launch } // stopped during init

                when (recognizer.checkStatus()) {
                    FeatureStatus.UNAVAILABLE -> {
                        onError("Speech recognition isn't available on this device.")
                        return@launch
                    }
                    FeatureStatus.DOWNLOADABLE, FeatureStatus.DOWNLOADING -> {
                        // One-time on-device model download. Collect to completion before listening.
                        recognizer.download().collect { }
                    }
                    else -> {} // AVAILABLE
                }
                if (stopRequested) { onFinal(""); return@launch } // stopped during download

                val request = speechRecognizerRequest { audioSource = AudioSource.fromMic() }
                listening = true
                recognizer.startRecognition(request).collect { response ->
                    when (response) {
                        is SpeechRecognizerResponse.PartialTextResponse -> onPartial(response.text)
                        is SpeechRecognizerResponse.FinalTextResponse -> onFinal(response.text)
                        is SpeechRecognizerResponse.ErrorResponse ->
                            onError(response.e.message ?: "Speech recognition error.")
                        else -> {} // CompletedResponse
                    }
                }
            } catch (c: CancellationException) {
                throw c // scope/job cancelled (e.g. close()) — not a user-facing error
            } catch (t: Throwable) {
                Log.e(TAG, "voice input failed", t)
                onError(t.message ?: "Voice input failed.")
            } finally {
                listening = false
                job = null
            }
        }
    }

    /**
     * Stop listening. If the recognizer is live, ask it to flush a final transcript (delivered via
     * the recognition flow). If we're still initializing / downloading, the `stopRequested` checks
     * in [start] bail with an empty final so the caller's state resets.
     */
    fun stop() {
        if (job == null) return
        stopRequested = true
        val recognizer = client
        if (listening && recognizer != null) {
            scope.launch { runCatching { recognizer.stopRecognition() } }
        }
    }

    fun close() {
        stopRequested = true
        listening = false
        job?.cancel()
        job = null
        runCatching { client?.close() }
        client = null
    }

    companion object {
        private const val TAG = "VoiceInput"
    }
}

package ai.zetic.aiberry.session

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import java.util.Locale

/**
 * Speaks each screening question aloud (Android TTS) and reports word-by-word
 * progress so the chat bubble can reveal the text in sync with the voice. The
 * Android parallel to iOS-Aiberry's `QuestionSpeaker`.
 *
 * Per question: [speak] → [onWord] fires as each word begins (with the substring
 * spoken so far, via `UtteranceProgressListener.onRangeStart`) → [onDone] fires once
 * the whole utterance is read, which is the cue to start recording the answer.
 *
 * All callbacks are delivered on the MAIN thread. If TTS init fails the speaker
 * degrades gracefully: it reveals the full text immediately and calls [onDone] so the
 * flow never stalls.
 */
class QuestionSpeaker(context: Context) {

    companion object {
        private const val TAG = "QuestionSpeaker"
        private const val UTTERANCE_ID = "screening-question"
    }

    private val main = Handler(Looper.getMainLooper())
    private var ready = false
    private var initFailed = false
    private var pending: (() -> Unit)? = null

    private var currentText = ""
    private var onWord: ((String) -> Unit)? = null
    private var onDone: (() -> Unit)? = null

    private val tts: TextToSpeech = TextToSpeech(context.applicationContext) { statusCode ->
        if (statusCode == TextToSpeech.SUCCESS) {
            tts.setLanguage(Locale.US)
            tts.setOnUtteranceProgressListener(progress)
            ready = true
            pending?.let { it() }
            pending = null
        } else {
            Log.w(TAG, "TTS init failed: $statusCode")
            initFailed = true
            // Flush any queued request so the flow doesn't hang.
            pending?.let { it() }
            pending = null
        }
    }

    private val progress = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) {}

        override fun onRangeStart(utteranceId: String?, start: Int, end: Int, frame: Int) {
            val text = currentText
            val upTo = end.coerceIn(0, text.length)
            main.post { onWord?.invoke(text.substring(0, upTo)) }
        }

        override fun onDone(utteranceId: String?) {
            main.post { finish() }
        }

        @Suppress("OVERRIDE_DEPRECATION") // abstract pre-API-21 overload; still required
        override fun onError(utteranceId: String?) {
            main.post { finish() }
        }

        override fun onError(utteranceId: String?, errorCode: Int) {
            main.post { finish() }
        }
    }

    /** Speak [text], revealing it word-by-word via [onWord], then call [onDone]. */
    fun speak(text: String, onWord: (String) -> Unit, onDone: () -> Unit) {
        this.currentText = text
        this.onWord = onWord
        this.onDone = onDone
        onWord("") // start from an empty bubble

        val request: () -> Unit = {
            if (initFailed) {
                // No TTS engine — reveal everything and continue immediately.
                onWord(text)
                finish()
            } else {
                tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
            }
        }
        if (ready || initFailed) request() else pending = request
    }

    /** Abandon any current speech without firing [onDone] (used on teardown). */
    fun cancel() {
        onWord = null
        onDone = null
        pending = null
        if (ready) runCatching { tts.stop() }
    }

    fun shutdown() {
        cancel()
        runCatching { tts.shutdown() }
    }

    private fun finish() {
        onWord?.invoke(currentText) // ensure full text is shown before recording
        val done = onDone
        onWord = null
        onDone = null
        done?.invoke()
    }
}

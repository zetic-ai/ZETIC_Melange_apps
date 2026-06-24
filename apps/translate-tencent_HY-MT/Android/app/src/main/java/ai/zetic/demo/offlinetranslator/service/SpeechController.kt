package ai.zetic.demo.offlinetranslator.service

import android.content.Context
import android.speech.tts.TextToSpeech
import ai.zetic.demo.offlinetranslator.model.Language
import java.util.Locale

/**
 * On-device (offline) text-to-speech — the Android equivalent of iOS `AVSpeechSynthesizer`.
 * Tapping a speaker button while speaking stops playback (stop-on-retap). TextToSpeech init is
 * asynchronous, so requests made before init completes are queued and flushed on ready.
 */
class SpeechController(context: Context) {
    private var ready = false
    private var pending: Pair<String, Locale>? = null

    private val tts = TextToSpeech(context.applicationContext) { status ->
        ready = status == TextToSpeech.SUCCESS
        if (ready) pending?.let { (text, locale) -> pending = null; speakNow(text, locale) }
    }

    /** Speak [text] in [language]'s voice. If already speaking, stop instead (toggle). */
    fun speak(text: String, language: Language) {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return

        if (ready && tts.isSpeaking) {
            tts.stop()
            return
        }

        val locale = bestLocale(language)
        if (ready) speakNow(trimmed, locale) else pending = trimmed to locale
    }

    private fun speakNow(text: String, locale: Locale) {
        // setLanguage falls back to the prefix (e.g. "ko") or default if the exact locale is absent.
        val result = tts.setLanguage(locale)
        if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
            tts.setLanguage(Locale(locale.language))
        }
        tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "offlinetranslator-utterance")
    }

    private fun bestLocale(language: Language): Locale = language.speechLocale

    fun shutdown() {
        runCatching { tts.stop() }
        runCatching { tts.shutdown() }
    }
}

package ai.zetic.demo.offlinetranslator.viewmodel

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.zetic.demo.offlinetranslator.model.Language
import ai.zetic.demo.offlinetranslator.ocr.ImageTextRecognizer
import ai.zetic.demo.offlinetranslator.service.NetworkMonitor
import ai.zetic.demo.offlinetranslator.service.SpeechController
import ai.zetic.demo.offlinetranslator.service.VoiceInputController
import ai.zetic.demo.offlinetranslator.translation.TranslationCleanup
import ai.zetic.demo.offlinetranslator.translation.TranslationPrompt
import ai.zetic.demo.offlinetranslator.translation.Translator
import ai.zetic.demo.offlinetranslator.translation.TranslatorFactory
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

sealed interface ModelState {
    data object Loading : ModelState
    data object Ready : ModelState
    data class Failed(val message: String) : ModelState
}

/**
 * Streaming translation state + on-device engine driver. Faithful Kotlin port of iOS
 * `TranslationViewModel`.
 *
 * Threading (critical): the ZeticMLange native model is blocking and NOT thread-safe — load, run,
 * waitForNextToken and cleanUp must all happen on ONE dedicated thread, never a thread pool (a
 * coroutine IO/Default dispatcher would crash native init). A single-thread executor serializes
 * every engine call. [gen] holds the latest generation id so an in-flight translation stops early
 * when superseded (mirrors the iOS `genLock`).
 */
class TranslationViewModel(app: Application) : AndroidViewModel(app) {

    // Editable translation state
    private val _sourceText = MutableStateFlow("")
    val sourceText: StateFlow<String> = _sourceText.asStateFlow()
    private val _translatedText = MutableStateFlow("")
    val translatedText: StateFlow<String> = _translatedText.asStateFlow()
    private val _source = MutableStateFlow(Language.detect)
    val source: StateFlow<Language> = _source.asStateFlow()
    private val _target = MutableStateFlow(Language.named("en"))
    val target: StateFlow<Language> = _target.asStateFlow()

    // Model + generation state
    private val _modelState = MutableStateFlow<ModelState>(ModelState.Loading)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()
    private val _downloadProgress = MutableStateFlow(0.0)
    val downloadProgress: StateFlow<Double> = _downloadProgress.asStateFlow()
    private val _isTranslating = MutableStateFlow(false)
    val isTranslating: StateFlow<Boolean> = _isTranslating.asStateFlow()

    // Voice / OCR input state
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()
    private val _partialVoiceText = MutableStateFlow("")
    val partialVoiceText: StateFlow<String> = _partialVoiceText.asStateFlow()
    private val _isRecognizingImage = MutableStateFlow(false)
    val isRecognizingImage: StateFlow<Boolean> = _isRecognizingImage.asStateFlow()
    private val _inputError = MutableStateFlow<String?>(null)
    val inputError: StateFlow<String?> = _inputError.asStateFlow()
    // One-shot signal: text captured from voice/OCR → screen should show the Result phase.
    private val _showResult = MutableSharedFlow<Unit>(extraBufferCapacity = 1)
    val showResult: SharedFlow<Unit> = _showResult.asSharedFlow()

    private val translator: Translator = TranslatorFactory.create(app)
    private val speech = SpeechController(app)
    private val voice = VoiceInputController(viewModelScope)
    private val networkMonitor = NetworkMonitor(app)
    val isOnline: StateFlow<Boolean> get() = networkMonitor.isOnline

    // ONE dedicated engine thread for every native call. Never a coroutine pool.
    private val engine = Executors.newSingleThreadExecutor { r ->
        Thread(r, "zetic-engine").apply { priority = Thread.NORM_PRIORITY + 1 }
    }
    private val gen = AtomicInteger(0)

    init {
        networkMonitor.start()
        loadModel()
    }

    // MARK: - Model lifecycle

    fun loadModel() {
        _modelState.value = ModelState.Loading
        _downloadProgress.value = 0.0
        Log.i(TAG, "Loading model ${translator::class.simpleName} …")
        engine.execute {
            try {
                translator.load { progress -> _downloadProgress.value = progress }
                Log.i(TAG, "Model ready")
                _modelState.value = ModelState.Ready
            } catch (t: Throwable) {
                Log.e(TAG, "Model load failed", t)
                _modelState.value = ModelState.Failed(t.message ?: "Failed to load the model.")
            }
        }
    }

    // MARK: - Translation

    fun canTranslate(): Boolean =
        _modelState.value == ModelState.Ready && _sourceText.value.trim().isNotEmpty()

    fun translate() {
        val text = _sourceText.value.trim()
        if (text.isEmpty() || _modelState.value != ModelState.Ready) return

        // New generation id supersedes any in-flight one (its callback sees a newer id and stops).
        val myGen = gen.incrementAndGet()
        _translatedText.value = ""
        _isTranslating.value = true

        val prompt = TranslationPrompt.make(text, _source.value, _target.value)
        engine.execute {
            translator.reset() // clear KV/conversation state from any prior turn
            val streamed = StringBuilder()
            try {
                translator.generate(prompt) { token ->
                    if (gen.get() != myGen) return@generate false // superseded → stop
                    streamed.append(token)
                    val snapshot = TranslationCleanup.clean(streamed.toString())
                    if (gen.get() == myGen) _translatedText.value = snapshot
                    true
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Translation failed", t)
                // A transient generation failure shouldn't tear down the loaded model (that's
                // reserved for load failures). Surface it as an input error and let the user
                // retry; the model stays Ready.
                if (gen.get() == myGen) _inputError.value = t.message ?: "Translation failed."
            }
            if (gen.get() == myGen) _isTranslating.value = false
        }
    }

    fun cancelTranslation() {
        gen.incrementAndGet() // supersede the in-flight generation
        _isTranslating.value = false
    }

    fun clearAll() {
        cancelTranslation()
        _sourceText.value = ""
        _translatedText.value = ""
    }

    // MARK: - Editable inputs

    fun setSourceText(value: String) {
        _sourceText.value = value
    }

    fun setSource(language: Language) {
        _source.value = language
    }

    fun setTarget(language: Language) {
        _target.value = language
    }

    // MARK: - Language controls

    fun swapLanguages() {
        val oldSource = _source.value
        val oldTarget = _target.value

        if (oldSource.isDetect) {
            // Can't make "Detect" a target: promote the target to source, pick a sensible target.
            _source.value = oldTarget
            _target.value = if (oldTarget.id == "en") Language.named("ko") else Language.named("en")
        } else {
            _source.value = oldTarget
            _target.value = oldSource
        }

        // Mirror DeepL: move the produced translation up into the input, then re-run.
        if (_translatedText.value.isNotEmpty()) {
            _sourceText.value = _translatedText.value
            _translatedText.value = ""
            translate()
        }
    }

    // MARK: - Actions (offline-friendly)

    fun paste() {
        val clipboard = getApplication<Application>()
            .getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val text = clipboard.primaryClip?.takeIf { it.itemCount > 0 }
            ?.getItemAt(0)?.coerceToText(getApplication())?.toString()
        if (!text.isNullOrEmpty()) _sourceText.value = text
    }

    fun copyTranslation() {
        val clipboard = getApplication<Application>()
            .getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("translation", _translatedText.value))
    }

    fun speak(text: String, language: Language) = speech.speak(text, language)

    // MARK: - Voice input (offline speech-to-text)

    /** Locale to recognize in: the source language, or the device locale when source is "Detect". */
    private fun recognitionLocale(): Locale =
        if (_source.value.isDetect) Locale.getDefault() else _source.value.speechLocale

    fun startVoiceInput() {
        if (_isListening.value) return
        _inputError.value = null
        _partialVoiceText.value = ""
        _isListening.value = true
        voice.start(
            locale = recognitionLocale(),
            onPartial = { _partialVoiceText.value = it },
            onFinal = { text -> finishVoice(text) },
            onError = { message ->
                _isListening.value = false
                _partialVoiceText.value = ""
                _inputError.value = message
            },
        )
    }

    /** Stop listening; a final transcript arrives via the controller's onFinal. */
    fun stopVoiceInput() = voice.stop()

    private fun finishVoice(text: String) {
        _isListening.value = false
        _partialVoiceText.value = ""
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return
        _sourceText.value = trimmed
        _showResult.tryEmit(Unit)
        translate()
    }

    // MARK: - Image OCR (offline)

    fun recognizeImage(uri: Uri) {
        _inputError.value = null
        _isRecognizingImage.value = true
        ImageTextRecognizer.recognize(
            context = getApplication(),
            uri = uri,
            language = _source.value,
            onResult = { text ->
                _isRecognizingImage.value = false
                val trimmed = text.trim()
                if (trimmed.isEmpty()) {
                    _inputError.value = "No text found in the image."
                    return@recognize
                }
                _sourceText.value = trimmed
                _showResult.tryEmit(Unit)
                translate()
            },
            onError = { message ->
                _isRecognizingImage.value = false
                _inputError.value = message
            },
        )
    }

    fun clearInputError() {
        _inputError.value = null
    }

    override fun onCleared() {
        super.onCleared()
        gen.incrementAndGet() // stop any in-flight generation
        engine.execute { translator.tearDown() } // release native model on the engine thread
        engine.shutdown()
        speech.shutdown()
        voice.close()
        networkMonitor.stop()
    }

    companion object {
        private const val TAG = "TranslationVM"
    }
}

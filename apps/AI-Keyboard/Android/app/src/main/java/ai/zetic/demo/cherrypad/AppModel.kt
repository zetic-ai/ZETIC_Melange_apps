package ai.zetic.demo.cherrypad

import android.app.Application
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.zetic.demo.cherrypad.data.Prefs
import ai.zetic.demo.cherrypad.llm.LLMService
import ai.zetic.demo.cherrypad.llm.Prompts
import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.model.Language
import ai.zetic.demo.cherrypad.model.Stance
import ai.zetic.demo.cherrypad.model.Tone
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The container-app state machine (mirrors iOS `AppModel`). Owns the compose-screen inputs
 * and drives generation through the shared [LLMService].
 */
class AppModel(app: Application) : AndroidViewModel(app) {

    var inputText by mutableStateOf("")
    var task by mutableStateOf(KeyboardTask.REWRITE)
    var tone by mutableStateOf(Tone.PROFESSIONAL)
    var stance by mutableStateOf(Stance.AGREEABLE)
        private set
    var targetLanguage by mutableStateOf(
        Language.matching(Prefs.targetLanguageName(app)) ?: Language.default
    )
        private set

    var resultText by mutableStateOf("")
        private set
    var isGenerating by mutableStateOf(false)
        private set
    var hasResult by mutableStateOf(false)
        private set
    var errorMessage by mutableStateOf<String?>(null)
        private set
    var didApply by mutableStateOf(false)
        private set

    val canGenerate: Boolean get() = inputText.isNotBlank() && !isGenerating

    private var genJob: Job? = null

    fun selectTone(value: Tone) { tone = value }
    fun selectStance(value: Stance) { stance = value }

    /** Setting the target language persists it so the keyboard's Translate uses the same target. */
    fun selectTargetLanguage(value: Language) {
        targetLanguage = value
        Prefs.setTargetLanguageName(getApplication(), value.englishName)
    }

    fun run() {
        val text = inputText.trim()
        if (text.isEmpty()) return
        genJob?.cancel()
        resultText = ""
        errorMessage = null
        didApply = false
        hasResult = true
        isGenerating = true

        val prompt = Prompts.build(task, text, tone, stance, targetLanguage.englishName)
        val maxTokens = Prompts.maxTokens(task)

        genJob = viewModelScope.launch {
            try {
                LLMService.ensureLoaded(getApplication())
                if (!LLMService.isReady) {
                    isGenerating = false
                    errorMessage = LLMService.loadError ?: "The model isn't ready yet."
                    return@launch
                }
                val finalText = LLMService.generateSanitized(prompt, maxTokens) { partial ->
                    resultText = partial
                }
                withContext(Dispatchers.Main) {
                    resultText = finalText
                    isGenerating = false
                }
            } catch (e: CancellationException) {
                isGenerating = false
                throw e
            } catch (e: Throwable) {
                isGenerating = false
                errorMessage = e.message ?: "Something went wrong."
            }
        }
    }

    fun retake() = run()

    fun cancel() {
        genJob?.cancel()
        LLMService.cancel()
        isGenerating = false
    }

    fun apply() {
        val text = resultText
        if (text.isBlank()) return
        val cm = getApplication<Application>()
            .getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        cm.setPrimaryClip(ClipData.newPlainText("CherryPad", text))
        didApply = true
    }
}

package ai.zetic.demo.cherrypad.keyboard

import ai.zetic.demo.cherrypad.data.Prefs
import ai.zetic.demo.cherrypad.llm.LLMService
import ai.zetic.demo.cherrypad.llm.Prompts
import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.model.Stance
import ai.zetic.demo.cherrypad.model.Tone
import ai.zetic.demo.cherrypad.ui.theme.CherryTheme
import android.inputmethodservice.InputMethodService
import android.view.View
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import androidx.compose.ui.platform.ComposeView
import androidx.compose.ui.platform.ViewCompositionStrategy
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry
import androidx.lifecycle.ViewModelStore
import androidx.lifecycle.ViewModelStoreOwner
import androidx.lifecycle.setViewTreeLifecycleOwner
import androidx.lifecycle.setViewTreeViewModelStoreOwner
import androidx.savedstate.SavedStateRegistry
import androidx.savedstate.SavedStateRegistryController
import androidx.savedstate.SavedStateRegistryOwner
import androidx.savedstate.setViewTreeSavedStateRegistryOwner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

/**
 * The CherryPad soft keyboard. Runs LFM2.5-350M in-process (via the shared [LLMService]) —
 * no app round-trip. Hosts a Compose keyboard view; because an [InputMethodService] is not a
 * lifecycle/viewmodel/savedstate owner, this service implements those so `ComposeView` works.
 */
class CherryImeService : InputMethodService(),
    LifecycleOwner, ViewModelStoreOwner, SavedStateRegistryOwner, KeyboardActions {

    private val lifecycleRegistry = LifecycleRegistry(this)
    override val lifecycle: Lifecycle get() = lifecycleRegistry

    override val viewModelStore = ViewModelStore()

    private val savedStateRegistryController = SavedStateRegistryController.create(this)
    override val savedStateRegistry: SavedStateRegistry
        get() = savedStateRegistryController.savedStateRegistry

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private val state = KeyboardState()

    override fun onCreate() {
        super.onCreate()
        savedStateRegistryController.performRestore(null)
        // Drive the lifecycle straight to RESUMED and keep it there for the service's life.
        // Compose's window recomposer only produces frames while its ViewTreeLifecycleOwner is
        // >= STARTED; the input view attaches/detaches as the keyboard shows/hides, but the
        // owner staying RESUMED guarantees the composition renders every time it re-attaches.
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_CREATE)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_START)
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_RESUME)
    }

    override fun onCreateInputView(): View {
        val view = ComposeView(this).apply {
            setViewCompositionStrategy(ViewCompositionStrategy.DisposeOnViewTreeLifecycleDestroyed)
            setContent {
                CherryTheme { KeyboardScreen(state, this@CherryImeService) }
            }
        }
        // The owners must be discoverable from the input view AND the IME window's decor view
        // (the recomposer is installed against the window root), or the composition never draws.
        window?.window?.decorView?.let { decor ->
            decor.setViewTreeLifecycleOwner(this)
            decor.setViewTreeViewModelStoreOwner(this)
            decor.setViewTreeSavedStateRegistryOwner(this)
        }
        view.setViewTreeLifecycleOwner(this)
        view.setViewTreeViewModelStoreOwner(this)
        view.setViewTreeSavedStateRegistryOwner(this)
        return view
    }

    override fun onStartInputView(info: EditorInfo?, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        state.banner = null
    }

    override fun onDestroy() {
        lifecycleRegistry.handleLifecycleEvent(Lifecycle.Event.ON_DESTROY)
        viewModelStore.clear()
        scope.cancel()
        super.onDestroy()
    }

    // ---- KeyboardActions ----

    override fun insert(text: String) {
        currentInputConnection?.commitText(text, 1)
        if (state.shifted && text != " ") state.shifted = false
    }

    override fun deleteBackward() {
        val ic = currentInputConnection ?: return
        val selected = ic.getSelectedText(0)
        if (!selected.isNullOrEmpty()) ic.commitText("", 1)
        else ic.deleteSurroundingText(1, 0)
    }

    override fun newLine() {
        currentInputConnection?.commitText("\n", 1)
    }

    override fun nextKeyboard() {
        val switched = runCatching { switchToNextInputMethod(false) }.getOrDefault(false)
        if (!switched) {
            (getSystemService(INPUT_METHOD_SERVICE) as? InputMethodManager)?.showInputMethodPicker()
        }
    }

    /** Capture the current selection, else the surrounding text. */
    private fun capturedText(): String {
        val ic = currentInputConnection ?: return ""
        val selected = ic.getSelectedText(0)
        if (!selected.isNullOrEmpty()) return selected.toString()
        val before = ic.getTextBeforeCursor(4000, 0)?.toString().orEmpty()
        val after = ic.getTextAfterCursor(4000, 0)?.toString().orEmpty()
        return (before + after).trim()
    }

    override fun runAction(task: KeyboardTask) {
        if (state.processing) return
        val text = capturedText()
        if (text.isBlank()) {
            state.banner = "Type or select text first, then tap again."
            return
        }
        state.banner = null
        state.resultText = null
        state.activeTask = task
        state.processing = true
        state.statusText = "Preparing…"

        val tone = if (task == KeyboardTask.REWRITE) Tone.PROFESSIONAL else null
        val stance = if (task == KeyboardTask.REPLY) Stance.AGREEABLE else null
        val lang = if (task == KeyboardTask.TRANSLATE) Prefs.targetLanguageName(this) else null
        val prompt = Prompts.build(task, text, tone, stance, lang)
        val maxTokens = Prompts.maxTokens(task)

        scope.launch {
            try {
                LLMService.ensureLoaded(applicationContext) { p ->
                    state.statusText =
                        if (p > 0f && p < 1f) "Downloading model… ${(p * 100).toInt()}%" else "Preparing…"
                }
                state.statusText = "Thinking…"
                val result = LLMService.generateSanitized(prompt, maxTokens)
                state.processing = false
                state.statusText = null
                if (result.isBlank()) state.banner = "No result — try again."
                else state.resultText = result
            } catch (e: Throwable) {
                state.processing = false
                state.statusText = null
                state.banner = "Failed: ${e.message ?: "error"}"
            }
        }
    }

    override fun insertResult() {
        val text = state.resultText ?: return
        // commitText replaces the active selection if any, else inserts at the cursor.
        currentInputConnection?.commitText(text, 1)
        state.resultText = null
        state.activeTask = null
    }

    override fun dismissResult() {
        state.resultText = null
        state.activeTask = null
        state.banner = null
    }
}

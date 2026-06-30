package com.brew.vm

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.brew.BrewApplication
import com.brew.data.NoteEntity
import com.brew.data.NoteStatus
import com.brew.llm.Prompts
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

data class EnhanceState(
    val generating: Boolean = false,
    val streamingText: String = "",
    val error: String? = null,
)

class NoteDetailViewModel(app: Application, val noteId: String) : AndroidViewModel(app) {
    private val appCtx = app as BrewApplication
    private val repo = appCtx.repository
    private val coordinator = appCtx.coordinator
    private val worker = appCtx.transcriptionWorker

    val note: StateFlow<NoteEntity?> =
        repo.observeNote(noteId)
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    val llmPhase = coordinator.llmPhase

    private val _enhance = MutableStateFlow(EnhanceState())
    val enhance: StateFlow<EnhanceState> = _enhance

    private var autoTriggered = false

    /** Auto-generate the AI note once the transcript is ready (iOS `.task`). */
    fun onNoteChanged(note: NoteEntity) {
        if (autoTriggered) return
        if (note.enhancedNote.isNullOrEmpty() &&
            note.noteStatus == NoteStatus.TRANSCRIBED &&
            note.transcript.isNotBlank()
        ) {
            autoTriggered = true
            generate()
        }
    }

    fun generate() {
        val current = note.value ?: return
        if (_enhance.value.generating) return
        _enhance.value = EnhanceState(generating = true)
        viewModelScope.launch {
            try {
                repo.updateStatus(noteId, NoteStatus.ENHANCING)
                val prompt = Prompts.enhance(current.transcript)
                val text = coordinator.withLlm { llm ->
                    llm.generateSanitized(prompt, maxTokens = 512) { partial ->
                        _enhance.value = _enhance.value.copy(streamingText = partial)
                    }
                }
                val title = deriveTitle(text)
                repo.setEnhancedNote(noteId, text, title)
                _enhance.value = EnhanceState(generating = false)
            } catch (t: Throwable) {
                repo.updateStatus(noteId, NoteStatus.TRANSCRIBED)
                _enhance.value = EnhanceState(
                    generating = false,
                    error = t.message ?: "Couldn't generate the note.",
                )
            }
        }
    }

    fun retryTranscription() {
        autoTriggered = false
        worker.enqueue(noteId)
    }

    /** Title = first `# ` heading, else first meaningful line, capped at 60. */
    private fun deriveTitle(enhanced: String): String {
        val heading = enhanced.lineSequence()
            .map { it.trimStart() }
            .firstOrNull { it.startsWith("# ") }
        if (heading != null) {
            return heading.removePrefix("# ").trim().take(60)
        }
        val line = enhanced.lineSequence()
            .map { it.trim() }
            .firstOrNull { it.isNotEmpty() && !it.startsWith("#") }
        return (line ?: "New note").take(60)
    }
}

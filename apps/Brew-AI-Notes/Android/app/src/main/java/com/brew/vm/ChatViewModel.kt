package com.brew.vm

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.brew.BrewApplication
import com.brew.data.ChatMessageEntity
import com.brew.data.ChatRole
import com.brew.llm.Prompts
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.util.Date

data class ChatUiState(
    val responding: Boolean = false,
    val streamingReply: String = "",
    val error: String? = null,
)

class ChatViewModel(app: Application, private val noteId: String) : AndroidViewModel(app) {
    private val appCtx = app as BrewApplication
    private val repo = appCtx.repository
    private val coordinator = appCtx.coordinator

    val messages: StateFlow<List<ChatMessageEntity>> =
        repo.observeMessages(noteId)
            .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    val llmPhase = coordinator.llmPhase

    private val _ui = MutableStateFlow(ChatUiState())
    val ui: StateFlow<ChatUiState> = _ui

    fun send(question: String) {
        val q = question.trim()
        if (q.isEmpty() || _ui.value.responding) return
        _ui.value = ChatUiState(responding = true)
        viewModelScope.launch {
            try {
                repo.addMessage(noteId, ChatRole.USER, q)
                val note = repo.getNote(noteId)
                val history = repo.getMessages(noteId).map {
                    Prompts.Turn(it.chatRole, it.content)
                }
                val context = Prompts.noteContext(
                    title = note?.title ?: "",
                    date = Date(note?.createdAt ?: System.currentTimeMillis()),
                    enhancedNote = note?.enhancedNote,
                    transcript = note?.transcript ?: "",
                )
                val prompt = Prompts.chatPrompt(Prompts.chatSystem(), context, history, q)
                val reply = coordinator.withLlm { llm ->
                    llm.generateSanitized(prompt, maxTokens = 320) { partial ->
                        _ui.value = _ui.value.copy(streamingReply = partial)
                    }
                }
                repo.addMessage(noteId, ChatRole.ASSISTANT, reply)
                _ui.value = ChatUiState()
            } catch (t: Throwable) {
                _ui.value = ChatUiState(error = t.message ?: "Something went wrong.")
            }
        }
    }
}

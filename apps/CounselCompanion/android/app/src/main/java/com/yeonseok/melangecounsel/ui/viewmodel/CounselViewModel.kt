package com.yeonseok.melangecounsel.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.yeonseok.melangecounsel.data.repository.ChatRepository
import com.yeonseok.melangecounsel.data.repository.DiagnosticsRepository
import com.yeonseok.melangecounsel.data.repository.SettingsRepository
import com.yeonseok.melangecounsel.domain.ChatMessageUi
import com.yeonseok.melangecounsel.domain.ChatSessionUi
import com.yeonseok.melangecounsel.domain.DiagnosticsSnapshot
import com.yeonseok.melangecounsel.domain.MessageRole
import com.yeonseok.melangecounsel.llm.ModelCompatibilityLayer
import com.yeonseok.melangecounsel.llm.ZeticChatEngine
import com.zeticai.mlange.core.model.ModelLoadingStatus
import com.yeonseok.melangecounsel.ui.theme.ThemeMode
import com.yeonseok.melangecounsel.util.ConnectivityObserver
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class CounselUiState(
    val sessions: List<ChatSessionUi> = emptyList(),
    val currentSessionId: Long? = null,
    val messages: List<ChatMessageUi> = emptyList(),
    val draftMessage: String = "",
    val isGenerating: Boolean = false,
    val isDownloading: Boolean = true,
    val downloadProgress: Float = 0f,
    val initializationState: String = "Checking Model...",
    val loadingMessage: String = "Ready",
    val networkStatus: String = "Unknown",
    val themeMode: String = ThemeMode.SYSTEM.name,
    val systemPrompt: String = "",
    val diagnostics: DiagnosticsSnapshot = DiagnosticsSnapshot(),
    val modelId: String = ZeticChatEngine.MODEL_ID,
    val maskedPersonalKey: String = ZeticChatEngine.maskedPersonalKey(),
    val errorMessage: String? = null
)

class CounselViewModel(
    private val chatRepository: ChatRepository,
    private val settingsRepository: SettingsRepository,
    private val diagnosticsRepository: DiagnosticsRepository,
    private val connectivityObserver: ConnectivityObserver,
    private val chatEngine: ZeticChatEngine
) : ViewModel() {

    private val compatibilityLayer = ModelCompatibilityLayer()

    private val _uiState = MutableStateFlow(CounselUiState())
    val uiState: StateFlow<CounselUiState> = _uiState.asStateFlow()

    private var generationJob: Job? = null
    private var messagesCollectionJob: Job? = null
    private var observedSessionId: Long? = null

    init {
        observeSessions()
        observeSettings()
        observeDiagnostics()
        observeConnectivity()
        viewModelScope.launch {
            if (_uiState.value.currentSessionId == null) {
                createNewSession()
            }
        }
        loadModel()
    }

    fun loadModel() {
        if (!_uiState.value.isDownloading) return
        viewModelScope.launch {
            try {
                chatEngine.initialize { status ->
                    _uiState.update {
                        it.copy(
                            initializationState = "Loading Model (${status.name})..."
                        )
                    }
                }
                _uiState.update { it.copy(isDownloading = false, initializationState = "Ready") }
            } catch (e: Exception) {
                _uiState.update {
                    it.copy(
                        isDownloading = false,
                        initializationState = "Model Error",
                        errorMessage = "Model initialization failed: ${e.message}"
                    )
                }
            }
        }
    }

    fun onDraftChanged(value: String) {
        _uiState.update { it.copy(draftMessage = value) }
    }

    fun createNewSession() {
        if (_uiState.value.isGenerating) return
        viewModelScope.launch {
            val sessionId = chatRepository.createSession("New Session ${System.currentTimeMillis()}")
            selectSession(sessionId)
        }
    }

    fun selectSession(sessionId: Long) {
        _uiState.update { it.copy(currentSessionId = sessionId) }
        observeMessagesForSession(sessionId)
    }

    fun sendMessage() {
        val message = _uiState.value.draftMessage.trim()
        val sessionId = _uiState.value.currentSessionId
        if (message.isEmpty() || _uiState.value.isGenerating || _uiState.value.isDownloading || sessionId == null) return
        if (!chatEngine.isInitialized) {
            _uiState.update { it.copy(errorMessage = "Model not initialized") }
            return
        }

        generationJob?.cancel()
        generationJob = viewModelScope.launch {
            val userMessageId = chatRepository.addMessage(sessionId, MessageRole.USER, message)
            if (userMessageId <= 0L) {
                _uiState.update { it.copy(errorMessage = "Failed to save your message.") }
                return@launch
            }

            _uiState.update {
                it.copy(
                    draftMessage = "",
                    isGenerating = true,
                    loadingMessage = "Loading Model (indeterminate)"
                )
            }

            val assistantMessageId = chatRepository.addMessage(sessionId, MessageRole.ASSISTANT, "")
            val history = chatRepository.getMessages(sessionId)
            val prompt = compatibilityLayer.buildPrompt(
                systemPrompt = _uiState.value.systemPrompt,
                history = if (history.size >= 2) history.dropLast(2) else emptyList(),
                userInput = message
            )

            val streamingBuffer = StringBuilder()
            val generation = chatEngine.generate(prompt = prompt) { token ->
                streamingBuffer.append(token)
                chatRepository.updateAssistantMessage(assistantMessageId, streamingBuffer.toString())
            }

            val stopReason = when {
                generation.errorMessage != null -> "error"
                generation.stopped -> "stopped"
                else -> "completed"
            }
            diagnosticsRepository.update(
                DiagnosticsSnapshot(
                    lastRunTimestamp = System.currentTimeMillis(),
                    lastGenerationMs = generation.durationMs,
                    lastTokenCount = generation.tokenCount,
                    lastRawLog = generation.fullText,
                    lastStopReason = stopReason
                )
            )

            _uiState.update {
                it.copy(
                    isGenerating = false,
                    loadingMessage = "Ready",
                    errorMessage = generation.errorMessage
                )
            }
        }
    }

    fun stopGeneration() {
        if (!_uiState.value.isGenerating) return
        chatEngine.requestStop()
        generationJob?.cancel()
        _uiState.update {
            it.copy(
                isGenerating = false,
                loadingMessage = "Generation stopped"
            )
        }
    }

    fun setThemeMode(themeMode: ThemeMode) {
        viewModelScope.launch {
            settingsRepository.setThemeMode(themeMode)
        }
    }

    fun setSystemPrompt(prompt: String) {
        viewModelScope.launch {
            settingsRepository.setSystemPrompt(prompt)
        }
    }

    fun clearHistory() {
        stopGeneration()
        viewModelScope.launch {
            chatRepository.clearAll()
            createNewSession()
        }
    }

    fun consumeError() {
        _uiState.update { it.copy(errorMessage = null) }
    }

    override fun onCleared() {
        stopGeneration()
        chatEngine.destroy()
        super.onCleared()
    }

    private fun observeSessions() {
        viewModelScope.launch {
            chatRepository.observeSessions().collect { sessions ->
                _uiState.update { current ->
                    val selectedId = when {
                        current.currentSessionId == null -> sessions.firstOrNull()?.id
                        sessions.any { it.id == current.currentSessionId } -> current.currentSessionId
                        else -> sessions.firstOrNull()?.id
                    }
                    current.copy(sessions = sessions, currentSessionId = selectedId)
                }
                val selectedId = _uiState.value.currentSessionId
                if (selectedId != null && selectedId != observedSessionId) {
                    observeMessagesForSession(selectedId)
                }
            }
        }
    }

    private fun observeMessagesForSession(sessionId: Long) {
        messagesCollectionJob?.cancel()
        observedSessionId = sessionId
        messagesCollectionJob = viewModelScope.launch {
            chatRepository.observeMessages(sessionId).collect { messages ->
                _uiState.update { it.copy(messages = messages) }
            }
        }
    }

    private fun observeSettings() {
        viewModelScope.launch {
            settingsRepository.themeModeFlow.collect { mode ->
                _uiState.update { it.copy(themeMode = mode.name) }
            }
        }
        viewModelScope.launch {
            settingsRepository.systemPromptFlow.collect { prompt ->
                _uiState.update { it.copy(systemPrompt = prompt) }
            }
        }
    }

    private fun observeDiagnostics() {
        viewModelScope.launch {
            diagnosticsRepository.snapshotFlow.collect { snapshot ->
                _uiState.update { it.copy(diagnostics = snapshot) }
            }
        }
    }

    private fun observeConnectivity() {
        viewModelScope.launch {
            connectivityObserver.observe().collect { status ->
                _uiState.update { it.copy(networkStatus = status) }
            }
        }
    }
}

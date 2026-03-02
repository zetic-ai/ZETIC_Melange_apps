package com.zeticai.qwen3chat.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.zeticai.qwen3chat.data.ChatRepository
import com.zeticai.qwen3chat.data.model.ChatMessage
import com.zeticai.qwen3chat.llm.LLMService
import com.zeticai.qwen3chat.llm.TokenSync
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private val chatRepository = ChatRepository(application)
    private val llmService = LLMService(application)
    
    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()
    
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()
    
    private val _isDownloading = MutableStateFlow(true)
    val isDownloading: StateFlow<Boolean> = _isDownloading.asStateFlow()
    
    private val _downloadProgress = MutableStateFlow(0f)
    val downloadProgress: StateFlow<Float> = _downloadProgress.asStateFlow()
    
    private val _initializationState = MutableStateFlow("Checking Model...")
    val initializationState: StateFlow<String> = _initializationState.asStateFlow()
    
    private val _currentStreamText = MutableStateFlow("")
    val currentStreamText: StateFlow<String> = _currentStreamText.asStateFlow()
    
    private val _lastGenerationTime = MutableStateFlow(0L)
    val lastGenerationTime: StateFlow<Long> = _lastGenerationTime.asStateFlow()
    
    private val _lastTokenCount = MutableStateFlow(0)
    val lastTokenCount: StateFlow<Int> = _lastTokenCount.asStateFlow()
    
    private var generationJob: Job? = null
    
    init {
        loadHistory()
    }
    
    fun loadModel() {
        if (!_isDownloading.value) return
        
        viewModelScope.launch {
            try {
                llmService.initialize { progress ->
                    _downloadProgress.value = progress
                    if (progress > 0.0f) {
                        _initializationState.value = "Downloading Model (${(progress * 100).toInt()}%)"
                    }
                }
                _isDownloading.value = false
            } catch (e: Exception) {
                _initializationState.value = "Model Error"
                _isDownloading.value = false
                val errorMsg = ChatMessage(isUser = false, text = "⚠️ Model initialization failed:\n${e.message}")
                val updatedList = _messages.value + errorMsg
                _messages.value = updatedList
            }
        }
    }
    
    private fun loadHistory() {
        viewModelScope.launch {
            _messages.value = chatRepository.loadMessages()
        }
    }
    
    private fun saveHistory(msgs: List<ChatMessage>) {
        viewModelScope.launch {
            chatRepository.saveMessages(msgs)
        }
    }
    
    fun sendMessage(text: String) {
        if (text.isBlank()) return
        
        val userMsg = ChatMessage(isUser = true, text = text)
        val updatedList = _messages.value + userMsg
        _messages.value = updatedList
        saveHistory(updatedList)
        
        // Simple Context Builder
        val prompt = buildPrompt(updatedList)
        
        _currentStreamText.value = ""
        _isGenerating.value = true
        
        generationJob = viewModelScope.launch {
            try {
                llmService.generateResponse(prompt).collect { sync ->
                    when (sync) {
                        is TokenSync.Token -> {
                            _currentStreamText.update { it + sync.text }
                        }
                        is TokenSync.Done -> {
                            finalizeResponse()
                            _lastGenerationTime.value = sync.durationMs
                            _lastTokenCount.value = sync.totalTokens
                        }
                    }
                }
            } catch (e: Exception) {
                _currentStreamText.value = "Error generating response."
                finalizeResponse()
            }
        }
    }
    
    private fun finalizeResponse() {
        if (_currentStreamText.value.isNotBlank()) {
            val aiMsg = ChatMessage(isUser = false, text = _currentStreamText.value)
            val updatedList = _messages.value + aiMsg
            _messages.value = updatedList
            saveHistory(updatedList)
        }
        _currentStreamText.value = ""
        _isGenerating.value = false
    }
    
    fun stopGeneration() {
        generationJob?.cancel()
        llmService.stop()
        finalizeResponse()
    }
    
    fun clearHistory() {
        viewModelScope.launch {
            chatRepository.clearHistory()
            _messages.value = emptyList()
        }
    }
    
    override fun onCleared() {
        super.onCleared()
        // Cleanup contract
        llmService.clear()
    }
    
    private fun buildPrompt(messages: List<ChatMessage>, maxCharacters: Int = 3000): String {
        var currentLength = 0
        val validMessages = mutableListOf<ChatMessage>()
        
        for (msg in messages.reversed()) {
            val role = if (msg.isUser) "User" else "Assistant"
            val line = "$role: ${msg.text}"
            if (currentLength + line.length > maxCharacters) {
                break
            }
            validMessages.add(0, msg)
            currentLength += line.length
        }
        
        return validMessages.joinToString("\n") { 
            if (it.isUser) "User: ${it.text}" else "Assistant: ${it.text}"
        } + "\nAssistant: "
    }
}

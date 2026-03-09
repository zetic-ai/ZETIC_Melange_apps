package com.yeonseok.melangecounsel.domain

import androidx.compose.runtime.Immutable

@Immutable
data class ChatMessageUi(
    val id: Long,
    val role: MessageRole,
    val content: String,
    val timestamp: Long,
    val isStreaming: Boolean = false
)

enum class MessageRole {
    USER,
    ASSISTANT
}

@Immutable
data class ChatSessionUi(
    val id: Long,
    val title: String,
    val updatedAt: Long,
    val preview: String
)

data class DiagnosticsSnapshot(
    val lastRunTimestamp: Long = 0L,
    val lastGenerationMs: Long = 0L,
    val lastTokenCount: Int = 0,
    val lastRawLog: String = "",
    val lastStopReason: String = "idle"
)

data class GenerationRecord(
    val fullText: String,
    val tokenCount: Int,
    val durationMs: Long,
    val stopped: Boolean,
    val errorMessage: String?
)

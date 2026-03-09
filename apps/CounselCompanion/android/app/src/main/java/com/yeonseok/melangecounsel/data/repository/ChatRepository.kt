package com.yeonseok.melangecounsel.data.repository

import com.yeonseok.melangecounsel.data.local.ChatDao
import com.yeonseok.melangecounsel.data.local.ChatMessageEntity
import com.yeonseok.melangecounsel.data.local.ChatSessionEntity
import com.yeonseok.melangecounsel.domain.ChatMessageUi
import com.yeonseok.melangecounsel.domain.ChatSessionUi
import com.yeonseok.melangecounsel.domain.MessageRole
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class ChatRepository(
    private val chatDao: ChatDao
) {
    fun observeSessions(): Flow<List<ChatSessionUi>> =
        chatDao.observeSessions().map { sessions ->
            val latestBySession = sessions.associate { session ->
                val preview = chatDao.getMessages(session.id).lastOrNull()?.content
                    ?.replace('\n', ' ')
                    ?.take(80)
                    .orEmpty()
                session.id to if (preview.isBlank()) "No messages yet" else preview
            }
            sessions.map { session ->
                ChatSessionUi(
                    id = session.id,
                    title = session.title,
                    updatedAt = session.updatedAt,
                    preview = latestBySession[session.id].orEmpty()
                )
            }
        }

    fun observeMessages(sessionId: Long): Flow<List<ChatMessageUi>> =
        chatDao.observeMessages(sessionId).map { list ->
            list.map { entity ->
                ChatMessageUi(
                    id = entity.id,
                    role = if (entity.role == "user") MessageRole.USER else MessageRole.ASSISTANT,
                    content = entity.content,
                    timestamp = entity.timestamp,
                    isStreaming = false
                )
            }
        }

    suspend fun getMessages(sessionId: Long): List<ChatMessageUi> {
        return chatDao.getMessages(sessionId).map { entity ->
            ChatMessageUi(
                id = entity.id,
                role = if (entity.role == "user") MessageRole.USER else MessageRole.ASSISTANT,
                content = entity.content,
                timestamp = entity.timestamp
            )
        }
    }

    suspend fun createSession(title: String): Long {
        val now = System.currentTimeMillis()
        return chatDao.insertSession(
            ChatSessionEntity(
                title = title,
                createdAt = now,
                updatedAt = now
            )
        )
    }

    suspend fun addMessage(sessionId: Long, role: MessageRole, content: String): Long {
        val now = System.currentTimeMillis()
        val messageId = chatDao.insertMessage(
            ChatMessageEntity(
                sessionId = sessionId,
                role = if (role == MessageRole.USER) "user" else "assistant",
                content = content,
                timestamp = now
            )
        )
        touchSession(sessionId, now, content)
        return messageId
    }

    suspend fun updateAssistantMessage(messageId: Long, content: String) {
        chatDao.updateMessageContent(messageId, content)
    }

    suspend fun clearAll() {
        chatDao.deleteAllMessages()
        chatDao.deleteAllSessions()
    }

    private suspend fun touchSession(sessionId: Long, timestamp: Long, content: String) {
        val session = chatDao.getSession(sessionId) ?: return
        val newTitle = if (session.title.startsWith("New Session")) {
            content.take(30).ifBlank { session.title }
        } else {
            session.title
        }
        chatDao.updateSession(
            session.copy(
                title = newTitle,
                updatedAt = timestamp
            )
        )
    }
}

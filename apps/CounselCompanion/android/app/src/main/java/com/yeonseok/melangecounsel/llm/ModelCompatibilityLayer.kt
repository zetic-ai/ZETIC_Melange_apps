package com.yeonseok.melangecounsel.llm

import com.yeonseok.melangecounsel.domain.ChatMessageUi
import com.yeonseok.melangecounsel.domain.MessageRole

class ModelCompatibilityLayer {
    fun buildPrompt(
        systemPrompt: String,
        history: List<ChatMessageUi>,
        userInput: String
    ): String {
        val builder = StringBuilder()
        if (systemPrompt.isNotBlank()) {
            builder.append("[System]\n")
            builder.append(systemPrompt.trim())
            builder.append("\n\n")
        }

        history.forEach { message ->
            when (message.role) {
                MessageRole.USER -> builder.append("[User] ")
                MessageRole.ASSISTANT -> builder.append("[Assistant] ")
            }
            builder.append(message.content.trim())
            builder.append("\n")
        }

        builder.append("[User] ")
        builder.append(userInput.trim())
        builder.append("\n[Assistant] ")

        return builder.toString().trim()
    }
}

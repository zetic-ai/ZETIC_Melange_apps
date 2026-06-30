package com.brew.llm

import com.brew.data.ChatRole
import java.text.DateFormat
import java.util.Date

/**
 * Builds the prompts handed to the local Gemma model (ported from iOS
 * `Prompts.swift`). Budgets are in characters, sized ~3 chars/token against the
 * 4096-token window. English only.
 */
object Prompts {
    private const val ENHANCED_NOTE_BUDGET = 3_000
    private const val TRANSCRIPT_BUDGET = 6_000
    private const val TRANSCRIPT_ONLY_BUDGET = 8_000
    private const val ENHANCE_TRANSCRIPT_BUDGET = 9_000
    private const val HISTORY_TURNS = 2
    private const val HISTORY_TURN_BUDGET = 400

    data class Turn(val role: ChatRole, val content: String)

    /** Keep the head (agenda/context) + tail (decisions/actions); drop the middle. */
    fun truncateMiddle(text: String, maxChars: Int): String {
        if (text.length <= maxChars) return text
        val headCount = maxChars * 6 / 10
        val tailCount = maxChars - headCount
        return "${text.take(headCount)}\n…[middle of transcript trimmed]…\n${text.takeLast(tailCount)}"
    }

    private fun trimmedHistory(history: List<Turn>, turns: Int): List<Turn> =
        history.takeLast(turns).map { Turn(it.role, it.content.take(HISTORY_TURN_BUDGET)) }

    private val todayString: String
        get() = DateFormat.getDateInstance(DateFormat.FULL).format(Date())

    private fun dateString(date: Date): String =
        DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(date)

    // --- Enhance (transcript -> structured markdown note) ---

    fun enhance(transcript: String): String {
        val system = """
            Turn the meeting transcript into clean, concise notes in English.
            Use Markdown: a few short `#` headings, each with 2-4 bullets.
            Capture key topics, decisions, and action items — brief, no filler, under ~300 words.
            Output only the notes: no preamble, reasoning, or commentary.
        """.trimIndent()

        val trimmed = truncateMiddle(transcript, ENHANCE_TRANSCRIPT_BUDGET)
        val user = "Transcript:\n" + trimmed.ifEmpty { "(no speech was transcribed)" }
        return compose(system, user)
    }

    // --- Chat (single note) ---

    fun chatSystem(): String = """
        You are Brew AI, a meeting assistant. Today is $todayString.
        Answer in English, in a few direct sentences — no preamble, no reasoning.
        Use only the provided transcript and summary; never invent details that are not present.
    """.trimIndent()

    fun noteContext(title: String, date: Date, enhancedNote: String?, transcript: String): String {
        val sb = StringBuilder("<context>\n")
        sb.append("Title: ").append(title.ifEmpty { "Untitled" }).append('\n')
        sb.append("Date: ").append(dateString(date)).append('\n')
        val hasSummary = !enhancedNote.isNullOrEmpty()
        if (hasSummary) {
            sb.append("Meeting Summary:\n")
                .append(enhancedNote!!.take(ENHANCED_NOTE_BUDGET)).append('\n')
        }
        val budget = if (hasSummary) TRANSCRIPT_BUDGET else TRANSCRIPT_ONLY_BUDGET
        val trimmed = truncateMiddle(transcript, budget)
        sb.append("Full Transcript:\n").append(trimmed.ifEmpty { "(empty)" }).append('\n')
        sb.append("</context>")
        return sb.toString()
    }

    /** Flattened single-string chat prompt: system + context + history + question. */
    fun chatPrompt(system: String, context: String, history: List<Turn>, question: String): String {
        val parts = mutableListOf(system, context)
        for (turn in trimmedHistory(history, HISTORY_TURNS)) {
            val speaker = if (turn.role == ChatRole.USER) "User" else "Assistant"
            parts.add("$speaker: ${turn.content}")
        }
        parts.add("User: $question")
        parts.add("Assistant:")
        return parts.joinToString("\n\n")
    }

    private fun compose(system: String, user: String): String = "$system\n\n---\n\n$user"
}

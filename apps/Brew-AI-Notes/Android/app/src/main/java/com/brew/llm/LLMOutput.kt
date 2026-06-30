package com.brew.llm

/**
 * Cleans raw local-model output for display (ported from iOS `LLMOutput`).
 *
 * Some models (harmony / channel style) emit reasoning before the answer, e.g.
 * `<|channel|>thought ... <|channel|>final ...`. We drop the reasoning segments
 * and strip the control tokens, keeping only user-facing content. It runs
 * incrementally, so on a growing stream it naturally hides the "thinking" until
 * the final answer begins.
 */
object LLMOutput {

    // Bracketed control tokens like `<|channel|>`, `<|channel>`, `<channel|>`, `<|message|>`.
    private val markerRegex = Regex(
        "<\\|?(?:channel|message|start|end|return|constrain|assistant|system|user|final|analysis|thought|commentary)\\|?>",
        RegexOption.IGNORE_CASE,
    )

    private val leadingLabelRegex = Regex(
        "^\\s*(?:final|message|assistant|channel)\\b[:\\-]?\\s*",
        RegexOption.IGNORE_CASE,
    )

    private val reasoningNames = listOf("thought", "analysis", "commentary")

    fun sanitize(raw: String): String {
        val sentinel = ""
        val replaced = markerRegex.replace(raw, sentinel)
        if (!replaced.contains(sentinel)) {
            return raw.trim()
        }

        val kept = ArrayList<String>()
        var prevWasReasoningName = false
        for (segment in replaced.split(sentinel)) {
            val lower = segment.trim().lowercase()
            if (lower.isEmpty()) continue

            if (prevWasReasoningName) {
                prevWasReasoningName = false
                continue
            }
            val name = reasoningNames.firstOrNull { lower.startsWith(it) }
            if (name != null) {
                prevWasReasoningName = (lower == name) // name-only → reasoning is in the next segment
                continue
            }
            val cleaned = leadingLabelRegex.replace(segment, "").trim()
            if (cleaned.isNotEmpty()) kept.add(cleaned)
        }
        return kept.joinToString("\n").trim()
    }
}

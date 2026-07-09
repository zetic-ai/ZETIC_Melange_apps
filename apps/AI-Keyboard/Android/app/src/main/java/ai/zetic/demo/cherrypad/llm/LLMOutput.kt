package ai.zetic.demo.cherrypad.llm

/**
 * Cleans raw local-model output for display. Runs incrementally on the growing stream,
 * so an in-progress reasoning span is hidden until the real answer begins.
 *
 * Order matters — mirrors the iOS `LLMOutput.sanitize` exactly.
 */
object LLMOutput {
    private val CLOSED_THINK = Regex("<think>[\\s\\S]*?</think>")
    private val CONTROL_TOKEN = Regex("<\\|[^>]*\\|>")
    private val ROLE_LABEL = Regex("^\\s*(?:assistant|system|user)\\s*[:\\n]", RegexOption.IGNORE_CASE)
    private val TASK_LABEL = Regex(
        "^\\s*(?:translated message|translated text|translation|rewritten message|rewritten text|rewrite|corrected message|corrected text|reply)\\s*:\\s*",
        RegexOption.IGNORE_CASE,
    )
    private val PREAMBLE = Regex(
        "^\\s*(?:sure[!,. ]*)?(?:here(?:'s| is)|okay|of course|certainly|absolutely)\\b[^:\\n]{0,60}:\\s*",
        RegexOption.IGNORE_CASE,
    )

    fun sanitize(raw: String): String {
        var s = raw

        // Drop any closed reasoning spans.
        s = CLOSED_THINK.replace(s, "")
        // If a reasoning span is still open mid-stream, hide from it onward.
        val open = s.indexOf("<think>")
        if (open >= 0) s = s.substring(0, open)
        // Strip ChatML / special control tokens like <|im_start|>, <|im_end|>.
        s = CONTROL_TOKEN.replace(s, "")
        // Some models echo a leading role label.
        s = ROLE_LABEL.replaceFirst(s, "")
        // Small models sometimes prepend a task label (e.g. "Translated message:").
        s = TASK_LABEL.replaceFirst(s, "")
        // …or a chatty meta-preamble ("Sure! Here's a quick reply:").
        s = PREAMBLE.replaceFirst(s, "")
        s = s.trim()
        // Strip symmetric wrapping quotes around the whole answer.
        if (s.length >= 2) {
            val first = s.first()
            val last = s.last()
            if ((first == '"' && last == '"') || (first == '“' && last == '”')) {
                s = s.substring(1, s.length - 1).trim()
            }
        }
        return s
    }
}

package ai.zetic.demo.cherrypad.llm

import ai.zetic.demo.cherrypad.model.KeyboardTask
import ai.zetic.demo.cherrypad.model.Stance
import ai.zetic.demo.cherrypad.model.Tone

/**
 * Builds the prompts for the four keyboard tasks.
 *
 * IMPORTANT: the ZeticMLange SDK applies the model's own chat template inside `run()`.
 * So we pass the instruction as plain user-message CONTENT — never raw ChatML
 * (`<|im_start|>` …) and never our own `User:`/`Assistant:` labels, both of which
 * confuse the model. Prompts are kept terse: the whole prompt is re-processed every
 * turn, so short prompts are the main latency lever.
 */
object Prompts {
    /** Cap input length — keyboard text is short, and runaway input wrecks latency. */
    private const val MAX_INPUT_CHARS = 1500

    /** Per-task output budgets (keyboard outputs are short). */
    fun maxTokens(task: KeyboardTask): Int = when (task) {
        KeyboardTask.GRAMMAR -> 256
        KeyboardTask.REWRITE -> 256
        KeyboardTask.REPLY -> 200
        KeyboardTask.TRANSLATE -> 320
    }

    /** Assembles the prompt for a request. */
    fun build(
        task: KeyboardTask,
        rawText: String,
        tone: Tone?,
        stance: Stance?,
        targetLanguage: String?,
    ): String {
        val text = rawText.take(MAX_INPUT_CHARS)
        return when (task) {
            KeyboardTask.REWRITE ->
                // Small models either collapse this (summarize) or echo it verbatim.
                // "Keep all the information; do not shorten" + a trailing prime gives
                // the most reliable full rewrite.
                compose(
                    instruction = "Rewrite this message in a ${tone?.descriptor ?: "clear"} tone. Keep all of its information and the same language; do not shorten or summarize.",
                    text = text,
                    prime = "Rewritten message:",
                )
            KeyboardTask.REPLY ->
                compose(
                    instruction = "Write a short, ${stance?.descriptor ?: "natural"} reply to the message below. Sound natural and human, in the same language as the message. Reply with only the reply text.",
                    text = text,
                )
            KeyboardTask.TRANSLATE -> {
                // Only the TARGET is given (the model infers the source).
                val lang = targetLanguage ?: "English"
                "Translate the following text into $lang, without additional explanation.\n\n$text"
            }
            KeyboardTask.GRAMMAR ->
                compose(
                    instruction = "Correct the grammar, spelling, and punctuation of the message below. Keep its meaning, tone, and language. If it is already correct, repeat it unchanged. Reply with only the corrected message.",
                    text = text,
                )
        }
    }

    /**
     * The instruction becomes the user-message content; the SDK wraps it in the model's
     * chat template. An optional [prime] primes the answer (e.g. keeps Rewrite from
     * collapsing).
     */
    private fun compose(instruction: String, text: String, prime: String? = null): String {
        var p = "$instruction\n\nMessage:\n$text"
        if (prime != null) p += "\n\n$prime"
        return p
    }
}

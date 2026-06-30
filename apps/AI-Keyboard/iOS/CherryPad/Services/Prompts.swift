import Foundation

/// Builds the prompts for the four keyboard tasks.
///
/// IMPORTANT: the ZeticMLange SDK applies the model's own chat template inside
/// `run(_:)` (verified on device — raw input comes back as an assistant turn). So
/// we pass the instruction as plain user-message CONTENT — never raw ChatML
/// (`<|im_start|>` …) and never our own `User:`/`Assistant:` labels, both of which
/// confuse the model. `/no_think` is appended on the fast (Qwen3) tier; without it
/// Qwen3-0.6B loops forever in `<think>` and produces no answer.
///
/// Prompts are kept terse: prefill cost is linear in prompt tokens and the whole
/// prompt is re-processed every turn, so short prompts are the main latency lever.
enum Prompts {
    /// Cap input length — keyboard text is short, and runaway input wrecks latency.
    private static let maxInputChars = 1500

    /// Per-task output budgets (keyboard outputs are short).
    static func maxTokens(for task: KeyboardTask) -> Int {
        switch task {
        case .grammar:   return 256
        case .rewrite:   return 256
        case .reply:     return 200
        case .translate: return 320
        }
    }

    /// Assembles the prompt for a request.
    static func build(task: KeyboardTask,
                      text rawText: String,
                      tone: Tone?,
                      stance: Stance?,
                      targetLanguage: String?) -> String {
        let text = String(rawText.prefix(maxInputChars))
        switch task {
        case .rewrite:
            // Small models either collapse this (summarize) or echo it verbatim.
            // "Keep all the information; do not shorten" + a trailing prime gives
            // the most reliable full rewrite.
            return compose(
                instruction: "Rewrite this message in a \(tone?.descriptor ?? "clear") tone. Keep all of its information and the same language; do not shorten or summarize.",
                text: text,
                prime: "Rewritten message:")
        case .reply:
            return compose(
                instruction: "Write a short, \(stance?.descriptor ?? "natural") reply to the message below. Sound natural and human, in the same language as the message. Reply with only the reply text.",
                text: text)
        case .translate:
            // Hunyuan-MT's official instruction phrasing — only the TARGET is given
            // (the model infers the source). Cleaner than a verbose instruction and
            // works better across small models too.
            let lang = targetLanguage ?? "English"
            let nt = ZeticConfig.quality.disablesThinking ? " /no_think" : ""
            return "Translate the following text into \(lang), without additional explanation.\(nt)\n\n\(text)"
        case .grammar:
            return compose(
                instruction: "Correct the grammar, spelling, and punctuation of the message below. Keep its meaning, tone, and language. If it is already correct, repeat it unchanged. Reply with only the corrected message.",
                text: text)
        }
    }

    /// The instruction becomes the user-message content; the SDK wraps it in the
    /// model's chat template. `/no_think` (fast/Qwen3 tier) prevents the reasoning
    /// loop; any stray `<think>` is stripped by `LLMOutput.sanitize`. An optional
    /// `prime` primes the answer (helps small models perform the task).
    private static func compose(instruction: String, text: String, prime: String? = nil) -> String {
        let noThink = ZeticConfig.quality.disablesThinking ? " /no_think" : ""
        var p = "\(instruction)\(noThink)\n\nMessage:\n\(text)"
        if let prime { p += "\n\n\(prime)" }
        return p
    }
}

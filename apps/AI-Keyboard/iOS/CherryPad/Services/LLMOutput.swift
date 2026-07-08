import Foundation

/// Cleans raw local-model output for display.
///
/// Qwen3 can emit a `<think>…</think>` reasoning span before the answer; with
/// `/no_think` it shouldn't, but this strips it as a safety net. ChatML control
/// tokens (`<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, …) are also removed.
/// Runs incrementally on the growing stream, so an in-progress reasoning span is
/// hidden until the real answer begins.
enum LLMOutput {
    static func sanitize(_ raw: String) -> String {
        var s = raw

        // Drop any closed reasoning spans.
        s = s.replacingOccurrences(
            of: #"<think>[\s\S]*?</think>"#,
            with: "",
            options: .regularExpression
        )
        // If a reasoning span is still open mid-stream, hide from it onward.
        if let open = s.range(of: "<think>") {
            s = String(s[..<open.lowerBound])
        }
        // Strip ChatML / special control tokens like <|im_start|>, <|im_end|>.
        s = s.replacingOccurrences(
            of: #"<\|[^>]*\|>"#,
            with: "",
            options: .regularExpression
        )
        // Some models echo a leading role label.
        s = s.replacingOccurrences(
            of: #"^\s*(?:assistant|system|user)\s*[:\n]"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        // Small models sometimes prepend a task label (e.g. "Translated message:").
        s = s.replacingOccurrences(
            of: #"^\s*(?:translated message|translated text|translation|rewritten message|rewritten text|rewrite|corrected message|corrected text|reply)\s*:\s*"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        // …or a chatty meta-preamble ("Sure! Here's a quick reply:", "Of course! Here is the translation:").
        s = s.replacingOccurrences(
            of: #"^\s*(?:sure[!,. ]*)?(?:here(?:'s| is)|okay|of course|certainly|absolutely)\b[^:\n]{0,60}:\s*"#,
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
        s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        // Strip symmetric wrapping quotes around the whole answer.
        if s.count >= 2, let first = s.first, let last = s.last,
           (first == "\"" && last == "\"") || (first == "“" && last == "”") {
            s = String(s.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }
}

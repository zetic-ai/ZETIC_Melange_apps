import Foundation

/// Cleans raw local-model output for display.
///
/// Some models (harmony / channel style) emit their reasoning before the answer,
/// e.g. `<|channel|>thought ... <|channel|>final ...`. We drop the reasoning
/// segments and strip the control tokens, keeping only user-facing content. It runs
/// incrementally, so when applied to a growing stream it naturally hides the
/// "thinking" until the final answer begins.
enum LLMOutput {
    private static let markerPattern =
        #"<\|?(?:channel|message|start|end|return|constrain|assistant|system|user|final|analysis|thought|commentary)\|?>"#

    private static let markerRegex = try? NSRegularExpression(pattern: markerPattern, options: [.caseInsensitive])

    private static let reasoningNames = ["thought", "analysis", "commentary"]

    static func sanitize(_ raw: String) -> String {
        let sentinel = "\u{0001}"
        guard let regex = markerRegex else {
            return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let ns = raw as NSString
        let replaced = regex.stringByReplacingMatches(
            in: raw,
            range: NSRange(location: 0, length: ns.length),
            withTemplate: sentinel
        )
        guard replaced.contains(sentinel) else {
            return raw.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        var kept: [String] = []
        var prevWasReasoningName = false
        for segment in replaced.components(separatedBy: sentinel) {
            let lower = segment.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if lower.isEmpty { continue }
            if prevWasReasoningName {
                prevWasReasoningName = false
                continue
            }
            if let name = reasoningNames.first(where: { lower.hasPrefix($0) }) {
                prevWasReasoningName = (lower == name)
                continue
            }
            let cleaned = segment.replacingOccurrences(
                of: #"^\s*(?:final|message|assistant|channel)\b[:\-]?\s*"#,
                with: "",
                options: [.regularExpression, .caseInsensitive]
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            if !cleaned.isEmpty { kept.append(cleaned) }
        }
        return kept.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

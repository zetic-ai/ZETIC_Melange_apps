import Foundation

struct PromptBuilder {
    func buildPrompt(systemPrompt: String, history: [Message], userInput: String, source: Language, target: Language) -> String {
        var lines: [String] = []
        let baseInstruction = "You are a precise translator using HY-MT. Translate the user text from \(source.name) to \(target.name). Respond with translation only, no explanations."
        lines.append(baseInstruction)
        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("System note: \(systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        lines.append("Source language: \(source.name) (\(source.code))")
        lines.append("Target language: \(target.name) (\(target.code))")
        lines.append("")
        for message in history {
            let roleLabel = message.role == .user ? "User" : "Assistant"
            lines.append("\(roleLabel): \(message.content.trimmingCharacters(in: .whitespacesAndNewlines))")
        }
        lines.append("User: \(userInput.trimmingCharacters(in: .whitespacesAndNewlines))")
        lines.append("Assistant:")
        return lines.joined(separator: "\n")
    }
}

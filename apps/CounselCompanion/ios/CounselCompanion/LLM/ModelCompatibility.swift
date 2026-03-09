import Foundation

struct ModelCompatibility {
    func buildPrompt(systemPrompt: String, history: [Message], userInput: String) -> String {
        var lines: [String] = []
        if !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines.append("[System]")
            lines.append(systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines))
            lines.append("")
        }
        for message in history {
            let prefix = message.role == .user ? "[User] " : "[Assistant] "
            lines.append(prefix + message.content.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        lines.append("[User] " + userInput.trimmingCharacters(in: .whitespacesAndNewlines))
        lines.append("[Assistant] ")
        return lines.joined(separator: "\n")
    }
}

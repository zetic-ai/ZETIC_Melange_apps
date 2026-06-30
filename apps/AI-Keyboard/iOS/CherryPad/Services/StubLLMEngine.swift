import Foundation

/// Placeholder engine used in the iOS Simulator, where the device-only
/// ZeticMLange framework can't run. It streams canned text so the full UI —
/// action bar, chips, result card, streaming — is exercisable without a device.
final class StubLLMEngine: LLMEngine {
    private var pending: [String] = []
    private var index = 0

    func load(onProgress: @escaping (Double) -> Void) throws {
        // Simulate a quick "download" so the progress UI is visible once.
        for step in stride(from: 0.0, through: 1.0, by: 0.25) {
            onProgress(step)
            Thread.sleep(forTimeInterval: 0.05)
        }
    }

    func startGeneration(prompt: String) throws -> Int {
        // Echo a plausible result for whichever task the prompt implies, so the
        // result card looks real in the Simulator.
        let lower = prompt.lowercased()
        let demo: String
        if lower.contains("translate") {
            demo = "Here is the simulated translation of your text into the selected language."
        } else if lower.contains("reply") {
            demo = "Sounds good to me — that works. Let me know what time suits you best!"
        } else if lower.contains("grammar") {
            demo = "This is the corrected sentence with grammar, spelling, and punctuation fixed."
        } else {
            demo = "Here is a polished rewrite of your message that keeps the same meaning."
        }
        pending = demo.split(separator: " ", omittingEmptySubsequences: false).map { $0 + " " }
        index = 0
        return prompt.count / 4
    }

    func nextToken() -> (token: String, isFinished: Bool) {
        guard index < pending.count else { return ("", true) }
        let token = pending[index]
        index += 1
        Thread.sleep(forTimeInterval: 0.03)
        return (token, false)
    }

    func stopGeneration() {
        pending = []
        index = 0
    }
}

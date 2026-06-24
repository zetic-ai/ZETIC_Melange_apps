import Foundation

/// Stand-in translator for the simulator/preview target, where the device-only
/// `ZeticMLange` SDK can't link. It streams a canned, plausible translation token by
/// token so the UI (progress, streaming result) behaves like the real thing.
final class MockTranslator: Translator {
    func load(onProgress: @escaping (Double) -> Void) throws {
        // TG_SLOW_LOAD stretches the fake download so the centered loading overlay
        // is observable in the simulator (preview/demo only).
        let step = ProcessInfo.processInfo.environment["TG_SLOW_LOAD"] != nil ? 0.35 : 0.03
        for i in 0...12 {                       // fake a model "download"
            onProgress(Double(i) / 12.0)
            Thread.sleep(forTimeInterval: step)
        }
    }

    func generate(prompt: String, onToken: (String) -> Bool) throws {
        let output = Self.cannedTranslation(for: prompt)
        for chunk in Self.tokenize(output) {
            Thread.sleep(forTimeInterval: 0.04) // mimic token latency so streaming is visible
            if !onToken(chunk) { return }
        }
    }

    func reset() {}
    func tearDown() {}

    // MARK: - Canned content

    private static func cannedTranslation(for prompt: String) -> String {
        let target = targetName(in: prompt).lowercased()
        let body = prompt.components(separatedBy: "\n\n").dropFirst().joined(separator: "\n\n")

        // A couple of nice demo outputs for the pitch screenshots.
        if target.contains("english") {
            return "Use Zetic to deploy your own AI model locally on any device."
        }
        if target.contains("korean") {
            return "Zetic을 사용하여 어떤 기기에서든 로컬로 나만의 AI 모델을 배포하세요."
        }
        if target.contains("japanese") {
            return "Zeticを使えば、あらゆるデバイス上でローカルに独自のAIモデルを展開できます。"
        }
        // Generic fallback that still demonstrates streaming.
        return "[\(target.capitalized) · offline] " + body
    }

    /// Pulls the target language name out of Hunyuan-MT's "...into X, ..." template
    /// (or the Chinese "翻译成…" template).
    private static func targetName(in prompt: String) -> String {
        let firstLine = prompt.components(separatedBy: "\n").first ?? prompt
        if let r = firstLine.range(of: "into ") {
            return firstLine[r.upperBound...]
                .prefix { $0 != "," && $0 != ":" }
                .trimmingCharacters(in: .whitespaces)
        }
        if firstLine.contains("翻译成") { return "chinese" }
        return "english"
    }

    /// Split into word-sized "tokens" (keeping trailing spaces) to mimic streaming.
    private static func tokenize(_ text: String) -> [String] {
        text.split(separator: " ", omittingEmptySubsequences: false)
            .enumerated()
            .map { $0.offset == 0 ? String($0.element) : " " + $0.element }
    }
}

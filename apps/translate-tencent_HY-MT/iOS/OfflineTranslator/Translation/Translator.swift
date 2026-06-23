import Foundation

enum TranslatorError: LocalizedError {
    case notLoaded

    var errorDescription: String? {
        switch self {
        case .notLoaded: return "The translation model isn't ready yet."
        }
    }
}

/// Abstraction over the on-device translation engine so the UI compiles in both the
/// device target (real `ZeticMLange` SDK) and the simulator/preview target (mock).
///
/// `load` and `generate` are **blocking** — the ZeticMLange implementation calls into
/// native code that must not run on the main thread. The caller (the ViewModel) owns
/// the off-main execution and the hop back to `@MainActor`.
protocol Translator: AnyObject {
    /// Prepare the model, reporting download progress in 0...1. Throws on failure.
    func load(onProgress: @escaping (Double) -> Void) throws

    /// Run one generation for `prompt`, invoking `onToken` for each produced token.
    /// `onToken` returns `false` to request an early stop (e.g. on cancellation).
    func generate(prompt: String, onToken: (String) -> Bool) throws

    /// Reset conversation / KV-cache state between translations.
    func reset()

    /// Fully release native resources.
    func tearDown()
}

/// Builds the prompt in Tencent **Hunyuan-MT**'s official instruction template. The model
/// was trained on these exact phrasings; a generic "Translate … to X" prompt makes it
/// unreliable (it sometimes echoes the source language instead of translating). Only the
/// *target* is specified — Hunyuan infers the source — so an explicit source selection is a
/// UI affordance and isn't part of the prompt.
enum TranslationPrompt {
    static func make(text: String, from source: Language, to target: Language) -> String {
        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if target.id == "zh-Hans" || target.id == "zh-Hant" {
            // Hunyuan-MT Chinese template, used when translating into Chinese (ZH<=>XX).
            return "把下面的文本翻译成\(chineseName(target))，不要额外解释。\n\n\(body)"
        }
        // Hunyuan-MT English template for every other target (XX<=>XX).
        return "Translate the following segment into \(englishName(target)), without additional explanation.\n\n\(body)"
    }

    /// "English (US)" → "English" for the prompt; other names are already clean.
    private static func englishName(_ language: Language) -> String {
        language.id == "en" ? "English" : language.englishName
    }

    private static func chineseName(_ language: Language) -> String {
        language.id == "zh-Hant" ? "繁体中文" : "简体中文"
    }
}

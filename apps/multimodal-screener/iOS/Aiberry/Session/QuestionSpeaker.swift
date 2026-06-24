import Foundation
import AVFoundation

/// Speaks each screening question aloud (Apple TTS) and reports word-by-word
/// progress so the chat bubble can reveal the text in sync with the voice.
///
/// Uses the shared application audio session (`usesApplicationAudioSession = true`,
/// the default) so it plays through the route `SessionAudio` already configured
/// (`.playAndRecord` + `.defaultToSpeaker`) without tearing it down mid-check-in.
///
/// Lifecycle per question: `speak(_:onWord:onFinish:)` →
///   `onWord(prefix)` fires as each word begins, with the substring spoken so far →
///   `onFinish()` fires once the whole utterance is read (or is cancelled), which is
///   the cue for the session to start recording the answer.
@MainActor
final class QuestionSpeaker: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    private let synth = AVSpeechSynthesizer()
    private var onWord: ((String) -> Void)?
    private var onFinish: (() -> Void)?
    private var spokenText = ""

    override init() {
        super.init()
        synth.delegate = self
    }

    /// Speak `text`, revealing it word-by-word via `onWord`, then call `onFinish`.
    func speak(_ text: String, onWord: @escaping (String) -> Void, onFinish: @escaping () -> Void) {
        // Drop any in-flight utterance and its callbacks first.
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
        self.spokenText = text
        self.onWord = onWord
        self.onFinish = onFinish

        onWord("")  // start from an empty bubble

        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        synth.speak(utterance)
    }

    /// Abandon any current speech without firing `onFinish` (used on teardown).
    func cancel() {
        onWord = nil
        onFinish = nil
        if synth.isSpeaking { synth.stopSpeaking(at: .immediate) }
    }

    // MARK: AVSpeechSynthesizerDelegate

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       willSpeakRangeOfSpeechString characterRange: NSRange,
                                       utterance: AVSpeechUtterance) {
        let end = characterRange.location + characterRange.length
        Task { @MainActor in
            guard self.synth === synthesizer else { return }
            let full = self.spokenText as NSString
            let upTo = min(end, full.length)
            self.onWord?(full.substring(to: upTo))
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            // Make sure the full text is shown before recording begins.
            self.onWord?(self.spokenText)
            let finish = self.onFinish
            self.onWord = nil
            self.onFinish = nil
            finish?()
        }
    }
}

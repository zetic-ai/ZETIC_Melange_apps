import Foundation
import SwiftUI
import AVFoundation
import os
#if canImport(UIKit)
import UIKit
#endif

@MainActor
final class TranslationViewModel: ObservableObject {
    // Editable translation state
    @Published var sourceText = ""
    @Published var translatedText = ""
    @Published var source: Language = .detect
    @Published var target: Language = .named("en")

    // Model + generation state
    @Published var modelState: ModelState = .loading
    @Published var downloadProgress: Double = 0
    @Published var isTranslating = false

    // Voice / OCR input state
    @Published var isListening = false
    @Published var partialVoiceText = ""
    @Published var isRecognizingImage = false
    @Published var inputError: String?
    /// Bumped when voice/OCR captures text, signalling the screen to show the result phase.
    @Published var showResultSignal = 0

    enum ModelState: Equatable {
        case loading
        case ready
        case failed(String)
    }

    private let translator: Translator
    private let synthesizer = AVSpeechSynthesizer()
    private let speechInput = SpeechInputController()

    // The ZeticMLange native model is blocking and not thread-safe: load + run +
    // waitForNextToken must all happen on ONE thread, never on Swift's cooperative
    // pool (Task.detached), which crashes the native init. A single dedicated serial
    // queue serializes every engine call — the pattern used by every working
    // ZeticMLange app. `genLock` holds the latest generation id so an in-flight
    // translation stops early when superseded.
    private let engineQueue = DispatchQueue(label: "ai.zetic.translate.engine", qos: .userInitiated)
    private let genLock = OSAllocatedUnfairLock(initialState: 0)
    private let log = Logger(subsystem: "ai.zetic.offlinetranslator", category: "model")

    var usingRealModel: Bool {
        #if canImport(ZeticMLange)
        return true
        #else
        return false
        #endif
    }

    init() {
        #if canImport(ZeticMLange)
        translator = ZeticTranslator()
        #else
        translator = MockTranslator()
        #endif
        loadModel()
    }

    // MARK: - Model lifecycle

    func loadModel() {
        modelState = .loading
        downloadProgress = 0
        let translator = self.translator
        let log = self.log
        log.info("Loading model \(ZeticConfig.modelName, privacy: .public) v\(ZeticConfig.modelVersion) …")
        engineQueue.async { [weak self] in
            do {
                try translator.load { progress in
                    DispatchQueue.main.async { self?.downloadProgress = progress }
                }
                log.info("Model ready")
                DispatchQueue.main.async { self?.modelState = .ready }
            } catch {
                let message = error.localizedDescription
                log.error("Model load failed: \(message, privacy: .public)")
                DispatchQueue.main.async { self?.modelState = .failed(message) }
            }
        }
    }

    // MARK: - Translation

    var canTranslate: Bool {
        modelState == .ready && !sourceText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func translate() {
        let text = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, modelState == .ready else { return }

        // New generation id supersedes any in-flight one (its callback sees a newer id and stops).
        let myGen = genLock.withLock { v -> Int in v += 1; return v }
        translatedText = ""
        isTranslating = true

        let prompt = TranslationPrompt.make(text: text, from: source, to: target)
        let translator = self.translator
        let genLock = self.genLock
        engineQueue.async { [weak self] in
            translator.reset() // clear KV/conversation state from any prior turn
            var streamed = ""
            do {
                try translator.generate(prompt: prompt) { token in
                    guard genLock.withLock({ $0 }) == myGen else { return false } // superseded → stop
                    streamed += token
                    let snapshot = TranslationCleanup.clean(streamed)
                    DispatchQueue.main.async {
                        guard let self, genLock.withLock({ $0 }) == myGen else { return }
                        self.translatedText = snapshot
                    }
                    return true
                }
            } catch {
                // A transient generation failure shouldn't tear down the loaded model
                // (that's reserved for load failures). Surface it as an input error and
                // let the user retry; the model stays `.ready`.
                let message = error.localizedDescription
                DispatchQueue.main.async {
                    guard let self, genLock.withLock({ $0 }) == myGen else { return }
                    self.inputError = message
                }
            }
            DispatchQueue.main.async {
                guard let self, genLock.withLock({ $0 }) == myGen else { return }
                self.isTranslating = false
            }
        }
    }

    func cancelTranslation() {
        genLock.withLock { $0 += 1 } // supersede the in-flight generation
        isTranslating = false
    }

    func clearAll() {
        cancelTranslation()
        sourceText = ""
        translatedText = ""
    }

    // MARK: - Language controls

    func swapLanguages() {
        let oldSource = source
        let oldTarget = target

        if oldSource.isDetect {
            // Can't make "Detect" a target: promote the target to source and pick a sensible target.
            source = oldTarget
            target = oldTarget.id == "en" ? .named("ko") : .named("en")
        } else {
            source = oldTarget
            target = oldSource
        }

        // Mirror DeepL: move the produced translation up into the input, then re-run.
        if !translatedText.isEmpty {
            sourceText = translatedText
            translatedText = ""
            translate()
        }
    }

    // MARK: - Actions (offline-friendly)

    func paste() {
        #if canImport(UIKit)
        if let string = UIPasteboard.general.string {
            sourceText = string
        }
        #endif
    }

    func copyTranslation() {
        #if canImport(UIKit)
        UIPasteboard.general.string = translatedText
        #endif
    }

    /// Reads `text` aloud with on-device (offline) text-to-speech. Tapping again while
    /// speaking stops it. Configures the audio session so it's actually audible on a
    /// real device, and falls back to any installed voice for the language.
    func speak(_ text: String, language: Language) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
            return
        }

        // Without an active .playback session, TTS is silent on device (e.g. in
        // silent mode) or routes oddly. spokenAudio + duckOthers is the right mode.
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)

        let utterance = AVSpeechUtterance(string: trimmed)
        utterance.voice = bestVoice(for: language)
        synthesizer.speak(utterance)
    }

    /// Exact voice for the language code, else any installed voice sharing its prefix
    /// (e.g. "ko-…"), else the system default.
    private func bestVoice(for language: Language) -> AVSpeechSynthesisVoice? {
        let code = language.speechCode
        if let exact = AVSpeechSynthesisVoice(language: code) { return exact }
        let prefix = String(code.prefix(2))
        return AVSpeechSynthesisVoice.speechVoices().first { $0.language.hasPrefix(prefix) }
    }

    // MARK: - Voice input (offline speech-to-text)

    /// Locale to recognize in: the source language, or the device locale when source is "Detect".
    private var recognitionLocale: Locale {
        source.isDetect ? Locale.current : Locale(identifier: source.speechCode)
    }

    func startVoiceInput() {
        guard !isListening else { return }
        inputError = nil
        partialVoiceText = ""
        isListening = true
        speechInput.start(
            locale: recognitionLocale,
            onPartial: { [weak self] text in self?.partialVoiceText = text },
            onFinal: { [weak self] text in self?.finishVoice(text) },
            onError: { [weak self] message in
                self?.isListening = false
                self?.partialVoiceText = ""
                self?.inputError = message
            }
        )
    }

    /// Stop listening; the recognizer flushes a final transcript via the controller's onFinal.
    func stopVoiceInput() {
        speechInput.stop()
    }

    private func finishVoice(_ text: String) {
        isListening = false
        partialVoiceText = ""
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        sourceText = trimmed
        showResultSignal += 1
        translate()
    }

    // MARK: - Image OCR (offline)

    func recognize(image: UIImage) {
        inputError = nil
        isRecognizingImage = true
        let languages = source.isDetect ? [] : [source.speechCode]
        VisionTextRecognizer.recognize(image, languages: languages) { [weak self] result in
            guard let self else { return }
            self.isRecognizingImage = false
            switch result {
            case .success(let text):
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    self.inputError = "No text found in the image."
                    return
                }
                self.sourceText = trimmed
                self.showResultSignal += 1
                self.translate()
            case .failure(let error):
                self.inputError = error.localizedDescription
            }
        }
    }

    deinit {
        genLock.withLock { $0 += 1 } // stop any in-flight generation
        let translator = self.translator
        engineQueue.async { translator.tearDown() } // release native model on the engine thread
    }
}

/// Trims artifacts from streamed model output (leading whitespace, an accidental
/// echo of the prompt's lead-in, surrounding quotes).
enum TranslationCleanup {
    static func clean(_ raw: String) -> String {
        var text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // If the model echoed the instruction line, drop it (Hunyuan rarely does, but be safe).
        for marker in ["Translate the following segment", "Translate the following text", "把下面的文本翻译成"] {
            if text.hasPrefix(marker), let newline = text.firstIndex(of: "\n") {
                text = String(text[text.index(after: newline)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return text
    }
}

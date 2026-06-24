import Foundation
import Speech
import AVFoundation

/// Offline speech-to-text via Apple's Speech framework. Uses on-device recognition
/// (`requiresOnDeviceRecognition`) plus an `AVAudioEngine` mic tap, so it works in Airplane Mode
/// once the language asset is installed. Small focused controller in the spirit of `NetworkMonitor`.
@MainActor
final class SpeechInputController {
    private let audioEngine = AVAudioEngine()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var delivered = false
    private var lastPartial = ""
    private var onFinalHandler: ((String) -> Void)?

    /// Intent flag: true from `start()` until a final is delivered or `stop()` finalizes.
    /// Kept separate from `capturing` (the audio engine actually running) so a `stop()` that
    /// lands during the async authorization window still cancels cleanly instead of being a no-op.
    private var active = false
    private var capturing = false

    /// Begins listening. `onPartial` streams interim text; `onFinal` delivers the final transcript;
    /// `onError` surfaces a user-facing message. Requests speech + mic authorization on first use.
    func start(
        locale: Locale,
        onPartial: @escaping (String) -> Void,
        onFinal: @escaping (String) -> Void,
        onError: @escaping (String) -> Void
    ) {
        guard !active else { return }
        active = true
        capturing = false
        delivered = false
        lastPartial = ""
        onFinalHandler = onFinal

        SFSpeechRecognizer.requestAuthorization { [weak self] auth in
            Task { @MainActor in
                guard let self, self.active else { return } // stopped during authorization
                guard auth == .authorized else {
                    self.failOut("Speech recognition permission is required.", onError); return
                }
                AVAudioSession.sharedInstance().requestRecordPermission { granted in
                    Task { @MainActor in
                        guard self.active else { return } // stopped during mic prompt
                        guard granted else {
                            self.failOut("Microphone access is required.", onError); return
                        }
                        self.begin(locale: locale, onPartial: onPartial, onFinal: onFinal, onError: onError)
                    }
                }
            }
        }
    }

    private func begin(
        locale: Locale,
        onPartial: @escaping (String) -> Void,
        onFinal: @escaping (String) -> Void,
        onError: @escaping (String) -> Void
    ) {
        guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
            failOut("Speech recognition isn't available for this language.", onError); return
        }
        self.recognizer = recognizer

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition {
            request.requiresOnDeviceRecognition = true
        }
        self.request = request

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(.record, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            failOut("Couldn't start audio: \(error.localizedDescription)", onError); return
        }

        let input = audioEngine.inputNode
        let format = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak request] buffer, _ in
            request?.append(buffer)
        }
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            cleanup()
            onError("Couldn't start the microphone.")
            return
        }

        capturing = true
        task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }
                if let result {
                    let text = result.bestTranscription.formattedString
                    if result.isFinal {
                        self.deliverFinal(text, onFinal)
                    } else {
                        self.lastPartial = text
                        onPartial(text)
                    }
                }
                if error != nil {
                    // A late error after the user stops is expected; only finalize once.
                    self.deliverFinal(self.lastPartial, onFinal)
                }
            }
        }
    }

    private func deliverFinal(_ text: String, _ onFinal: @escaping (String) -> Void) {
        guard !delivered else { return }
        delivered = true
        cleanup()
        onFinal(text)
    }

    /// Tears down after a pre-capture failure (bad locale, denied mic, audio-session error) and
    /// surfaces the message. Resets the intent flag so a later `start()` works.
    private func failOut(_ message: String, _ onError: @escaping (String) -> Void) {
        active = false
        onFinalHandler = nil
        onError(message)
    }

    /// User-initiated stop. If the recognizer is live, end audio so it flushes a final (delivered
    /// via the task handler). If we're still authorizing/initializing (not yet capturing), finalize
    /// now with whatever we have so the caller's state resets instead of getting stuck "listening".
    func stop() {
        guard active else { return }
        if capturing {
            if audioEngine.isRunning {
                audioEngine.stop()
                audioEngine.inputNode.removeTap(onBus: 0)
            }
            request?.endAudio()
        } else {
            deliverFinal(lastPartial, onFinalHandler ?? { _ in })
        }
    }

    private func cleanup() {
        if audioEngine.isRunning {
            audioEngine.stop()
            audioEngine.inputNode.removeTap(onBus: 0)
        }
        task?.cancel()
        task = nil
        request = nil
        capturing = false
        active = false
        onFinalHandler = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}

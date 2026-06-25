import Foundation
import Speech
import AVFoundation

/// On-device speech-to-text via Apple's `SFSpeechRecognizer` with
/// `requiresOnDeviceRecognition = true` — free, fully local, nothing sent to a
/// server. Powers the Transcript tab without needing a Melange ASR model.
final class SpeechTranscriber {
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))

    /// Request authorization once (called at launch alongside model preload).
    static func requestAuthorization() {
        SFSpeechRecognizer.requestAuthorization { _ in }
    }

    /// True only if on-device recognition is actually available on this device/locale.
    var available: Bool {
        guard let r = recognizer else { return false }
        return r.isAvailable && r.supportsOnDeviceRecognition
    }

    /// Transcribe several clips **sequentially** and return all transcripts in order.
    /// Sequential matters: firing multiple `recognitionTask`s on one recognizer at
    /// once makes the earlier ones fail/cancel — which previously left only the last
    /// answer transcribed. One-at-a-time keeps every answer.
    func transcribeAll(_ clips: [[Float]], completion: @escaping ([String]) -> Void) {
        var results = [String](repeating: "", count: clips.count)
        func step(_ i: Int) {
            guard i < clips.count else { completion(results); return }
            transcribe(clips[i]) { text in
                results[i] = text
                step(i + 1)
            }
        }
        step(0)
    }

    /// Transcribe a 16 kHz mono Float32 clip on-device. Always calls `completion`
    /// (with "" on failure/empty) so the caller's group balance is preserved.
    func transcribe(_ samples: [Float], completion: @escaping (String) -> Void) {
        guard available, !samples.isEmpty,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                         sampleRate: Double(AppConfig.sampleRate),
                                         channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count)) else {
            completion("")
            return
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let ch = buffer.floatChannelData?[0] {
            samples.withUnsafeBufferPointer { ch.update(from: $0.baseAddress!, count: samples.count) }
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        request.append(buffer)
        request.endAudio()

        var finished = false
        recognizer?.recognitionTask(with: request) { result, error in
            if let result, result.isFinal {
                finished = true
                completion(result.bestTranscription.formattedString)
            } else if error != nil, !finished {
                finished = true
                completion("")
            }
        }
    }
}

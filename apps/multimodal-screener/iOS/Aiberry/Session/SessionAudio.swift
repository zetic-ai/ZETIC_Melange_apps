import AVFoundation

/// Single owner of the `AVAudioSession` for the duration of a check-in.
///
/// A live `AVCaptureSession` (camera) plus `AVAudioEngine` (mic) coexist cleanly
/// only if the audio session is configured once and kept active — per-clip
/// activate/deactivate (the standalone `AudioRecorder` behavior) can glitch the
/// capture route. So during a check-in the recorder runs with `ownsSession = false`
/// and this owns the lifecycle. Mode `.videoRecording` routes/AGC sensibly while a
/// camera is running (better than `.measurement` here).
final class SessionAudio {
    private let session = AVAudioSession.sharedInstance()
    private(set) var active = false

    func begin() {
        guard !active else { return }
        try? session.setCategory(.playAndRecord, mode: .videoRecording,
                                 options: [.defaultToSpeaker, .allowBluetooth])
        try? session.setActive(true, options: [])
        try? session.overrideOutputAudioPort(.speaker)
        active = true
    }

    func end() {
        guard active else { return }
        try? session.setActive(false, options: [.notifyOthersOnDeactivation])
        active = false
    }
}

import AVFoundation
import Combine

/// Plays a bundled sample clip aloud through the speaker. Intended use: start a
/// recording, play a sample so the microphone picks it up, then stop the recording
/// to analyze it — an acoustic loopback through the live mic pipeline.
///
/// Deliberately does **not** touch the `AVAudioSession` category: while recording,
/// `AudioRecorder` owns a `.playAndRecord` session and this playback routes to the
/// speaker for the mic to hear; when idle, it plays via the default session.
final class SamplePlayer: NSObject, ObservableObject {
    /// Resource name of the clip currently playing, or nil when idle.
    @Published var playing: String?

    private var player: AVAudioPlayer?

    /// Play a bundled resource aloud. `key` identifies it for the `playing` state
    /// (defaults to the resource name). No-op if the resource can't be loaded.
    func play(resource name: String, ext: String, key: String? = nil) {
        guard let url = Bundle.main.url(forResource: name, withExtension: ext),
              let p = try? AVAudioPlayer(contentsOf: url) else { return }
        player?.stop()
        p.delegate = self
        player = p
        playing = key ?? name
        p.prepareToPlay()
        p.play()
    }

    func stop() {
        player?.stop()
        player = nil
        playing = nil
    }
}

extension SamplePlayer: AVAudioPlayerDelegate {
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async { self.player = nil; self.playing = nil }
    }

    func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
        DispatchQueue.main.async { self.player = nil; self.playing = nil }
    }
}

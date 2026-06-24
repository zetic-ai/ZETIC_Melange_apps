import Foundation

/// Central configuration for the demo.
///
/// Every model below runs **fully on-device** through ZETIC Melange — the
/// microphone audio never leaves the phone. Swap any `name` for a client's own
/// Melange model and the rest of the app keeps working unchanged: that is the
/// whole pitch.
enum AppConfig {
    /// ZETIC Melange Personal Access Key (dev key supplied by ZETIC for this demo).
    /// Replace with your own from https://mlange.zetic.ai → Settings.
    static let personalKey = "YOUR_MLANGE_KEY"

    /// Melange model identifiers (already hosted / uploaded).
    enum Model {
        /// 7-class wav2vec2-large-xlsr SER, uploaded as **version 2** of this repo.
        static let emotion = "realtonypark/Wav2Vec2-Base_Emotion-Recognition"
        static let emotionVersion = 2
        static let yamnet  = "google/Sound Classification(YAMNET)"
    }

    /// Audio capture settings shared by every tab.
    static let sampleRate = 16_000
    static let clipSeconds = 3.0
    static var clipSamples: Int { Int(Double(sampleRate) * clipSeconds) } // 48_000
}

/// Lifecycle of an on-device model, surfaced to the UI.
enum ModelStatus: Equatable {
    case idle
    case downloading(Float)   // 0…1, first run only (model is cached afterwards)
    case loading
    case running
    case ready
    case failed(String)

    var isBusy: Bool {
        switch self {
        case .downloading, .loading, .running: return true
        default: return false
        }
    }
}

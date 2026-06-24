import Foundation

/// Central configuration for the Aiberry "Berry Check-in" demo.
///
/// Every model below runs **fully on-device** through ZETIC Melange — the camera
/// frames and microphone audio never leave the phone. Swap any `name` for a
/// client's own Melange model and the rest of the app keeps working unchanged:
/// that is the whole pitch.
enum AppConfig {
    /// ZETIC Melange Personal Access Key (dev key supplied by ZETIC for this demo).
    /// Replace with your own from https://mlange.zetic.ai → Settings.
    static let personalKey = "YOUR_MLANGE_KEY"

    /// Melange model identifiers (already hosted / uploaded).
    enum Model {
        /// 7-class wav2vec2 speech-emotion recognition (voice modality). Reused as-is.
        static let emotion = "realtonypark/Wav2Vec2-Base_Emotion-Recognition"
        static let emotionVersion = 2

        /// Face-emotion recognition (face modality) — Elena Ryumina's ResNet50/AffectNet
        /// model, already hosted on Melange. Input `1×3×224×224` NCHW Float32, **BGR**,
        /// 0–255 with per-channel mean subtraction (no /255, no std). Output: 7 raw
        /// logits in `faceModelLabels` order. Preprocessing is done in Swift
        /// (`FacePixelTensor.bgrMeanSubtracted`), mapped to the canonical labels.
        static let face = "ElenaRyumina/FaceEmotionRecognition"
        static let faceVersion = 1
    }

    /// Canonical 7 emotion labels, shared by the voice + face models and the fusion
    /// engine. Per-model label *order* is read from each model; everything else maps
    /// by these strings so a model re-ordering its classes can't silently corrupt fusion.
    static let emotionLabels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    /// The hosted face model's native output order (Elena Ryumina / AffectNet), mapped
    /// to the canonical labels above so fusion/styling/avatar stay model-agnostic.
    static let faceModelLabels = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
    static let faceLabelToCanonical: [String: String] = [
        "Neutral": "Neutral", "Happiness": "Happy", "Sadness": "Sad",
        "Surprise": "Surprise", "Fear": "Fear", "Disgust": "Disgust", "Anger": "Angry",
    ]

    // MARK: Audio capture

    static let sampleRate = 16_000
    static let clipSeconds = 3.0
    static var clipSamples: Int { Int(Double(sampleRate) * clipSeconds) } // 48_000

    // MARK: Face capture

    enum Face {
        /// FER model input is a square `1 × 3 × size × size` tensor.
        static let inputSize = 224
        /// VGGFace2/Caffe-style BGR means subtracted from raw 0–255 pixels (no std).
        static let bgrMean: (b: Float, g: Float, r: Float) = (91.4953, 103.8827, 131.0912)
        /// Cap on-device FER cadence (Hz) so the ANE/thermals stay sane over a multi-
        /// minute session. One inference in flight at a time on top of this.
        static let inferenceHz: Double = 3
        static var frameInterval: Double { 1.0 / inferenceHz }
        /// Expand the Vision face box by this fraction before the square crop, so the
        /// FER model sees forehead/chin context it was trained on.
        static let cropMargin: CGFloat = 0.30
        /// Good frames needed before the face signal is fully trusted (drives confidence
        /// and the face/voice fusion weight). ~10 s at 3 Hz.
        static let targetFrames = 30
    }

    // MARK: Guided check-in

    enum CheckIn {
        /// Open-ended prompts the Berry avatar asks. Demo wording — not a clinical instrument.
        static let questions = [
            "How have you been feeling lately?",
            "What's been on your mind this week?",
            "Tell me about something that lifted or weighed on you recently.",
        ]
        /// Per-question capture bounds. The user can tap "Next" any time after `minSeconds`;
        /// capture auto-advances at `maxSeconds`.
        static let minSeconds: Double = 5
        static let maxSeconds: Double = 40
    }

    // MARK: Fusion (transparent, tunable)

    /// Russell circumplex coordinates per emotion (range −1…+1). Mood comes from
    /// valence, Energy from arousal. Kept here so the scoring is visible and tweakable.
    enum Affect {
        static let valence: [String: Float] = [
            "Angry": -0.7, "Disgust": -0.6, "Fear": -0.6, "Happy": 0.9,
            "Neutral": 0.0, "Sad": -0.8, "Surprise": 0.2,
        ]
        static let arousal: [String: Float] = [
            "Angry": 0.8, "Disgust": 0.3, "Fear": 0.8, "Happy": 0.6,
            "Neutral": 0.0, "Sad": -0.5, "Surprise": 0.8,
        ]
    }

    enum Fusion {
        /// Face/voice blend weight bounds (weight on face). More good face frames → more
        /// trust in the face stream; clamped so neither modality is ever fully ignored.
        static let faceWeightMin: Float = 0.2
        static let faceWeightMax: Float = 0.8
        /// Composite well-being weights (sum to 1): mood is monotonic; energy and speech
        /// rate are scored toward a healthy mid-band rather than "more is better".
        static let moodWeight: Float = 0.55
        static let energyWeight: Float = 0.30
        static let rateWeight: Float = 0.15
        /// Mid-band centres for the triangular preference of Energy / Rate (0…100 scale).
        static let energyIdeal: Float = 55
        static let rateIdeal: Float = 60
        /// Voiced fraction (speaking time / total) that maps to a "typical" speaking rate.
        static let typicalVoicedFraction: Float = 0.6
    }
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

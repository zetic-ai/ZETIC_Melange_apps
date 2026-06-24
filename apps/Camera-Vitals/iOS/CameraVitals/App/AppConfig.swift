import CoreGraphics
import Foundation

/// Central configuration & tuning constants for the rPPG pipeline.
enum AppConfig {
    // Melange model. Re-upload the low-memory bundle (melange/out_w30) to this model;
    // version `nil` always pulls the LATEST version, so no code change is needed after upload.
    static let personalKey = "YOUR_MLANGE_KEY"
    static let modelName = "realtonypark/EfficientPhys-rPPG_camera_vitals"
    static let modelVersion: Int? = nil   // nil = latest

    // Model I/O contract (small window to keep the per-inference memory peak low).
    // Each inference processes `framesIn` frames and emits `chunk` rPPG samples, which are
    // STITCHED into a rolling buffer for analysis (see analysisSamples).
    static let imgSize = 72
    static let channels = 3
    static let chunk = 30         // model output length (rPPG samples per inference)
    static let framesIn = 31      // model input frames (chunk + 1; model does torch.diff internally)

    // Capture / cadence. stride == chunk → consecutive windows are contiguous (1-frame overlap),
    // so their outputs stitch into a seamless waveform.
    static let fps: Double = 30
    static let stride = 30        // run inference every 30 new frames (~1 Hz)

    // Rolling analysis buffer of stitched rPPG samples used for the HR FFT.
    static let analysisSamples = 300       // 10 s @ 30 fps
    static let minAnalysisSamples = 150    // 5 s before the first heart-rate reading

    // Heart-rate band.
    static let hrBandLow = 0.75   // Hz  (45 bpm)
    static let hrBandHigh = 2.5   // Hz  (150 bpm)
    static let minBPM = 40.0
    static let maxBPM = 180.0

    // Quality gating.
    static let displayQualityFloor = 0.15   // below this, don't refresh the shown BPM
    static let lowLumaThreshold: Float = 40  // mean ROI luma (0-255) below this => "more light"

    // Guided measurement.
    static let measureDuration: TimeInterval = 30

    static var frameFloatCount: Int { channels * imgSize * imgSize }    // 15552
    static var inputFloatCount: Int { framesIn * frameFloatCount }      // 2_814_912
}

import Foundation

/// Top-level finite-state machine that drives which screen is shown.
enum MeasurementState: Equatable {
    case loadingModel(progress: Float)
    case permissionDenied
    case warmup(framesFilled: Int)   // buffer filling for the first time
    case live                        // continuous readout available
    case error(message: String)
}

/// Result of a guided 30-second measurement.
/// `Identifiable` with a stable per-instance `id` so the report sheet keeps a constant identity
/// across the frequent live-update re-renders behind it (otherwise the sheet flickers).
struct MeasurementReport: Equatable, Identifiable {
    let id = UUID()
    let avgBPM: Int
    let minBPM: Int
    let maxBPM: Int
    let avgQuality: Double
    let series: [Double]
}

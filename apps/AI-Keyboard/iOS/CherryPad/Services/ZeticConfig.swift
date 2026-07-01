import Foundation

/// Configuration for the ZETIC.ai Melange on-device model.
///
/// CherryPad ships a single model — **LFM2.5-350M** (Liquid Foundation Model): a
/// small, non-reasoning instruct model that runs entirely on-device (including
/// inside the keyboard extension). `personalKey` is the placeholder Melange token —
/// run `./adapt_mlange_key.sh` from the repo root to inject the real key.
enum ZeticConfig {
    static let personalKey = "dev_40e3948ba051485c9ccd827a2a17922f"
    static let modelName = "Steve/LFM2.5_350M"
    static let modelVersion = 1

    /// LFM2.5's dashboard recipe recommends RUN_ACCURACY.
    static let usesAccuracyMode = true
}

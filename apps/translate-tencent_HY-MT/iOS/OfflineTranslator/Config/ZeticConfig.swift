import Foundation

/// Configuration for the ZETIC.ai Melange on-device model.
///
/// `personalKey` is the placeholder Melange Personal Access Token — run
/// `./adapt_mlange_key.sh` from the repo root to fill it in. `modelName`/`version`
/// point at the Tencent HY-MT (Hunyuan-MT) model uploaded to the Melange dashboard.
enum ZeticConfig {
    static let personalKey = "YOUR_MLANGE_KEY"
    static let modelName = "vaibhav-zetic/tencent_HY-MT"
    static let modelVersion = 1
}

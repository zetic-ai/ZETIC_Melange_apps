import Foundation
import SwiftUI

/// Owns the three on-device models for the app's lifetime and preloads them at
/// launch so the NPU models are downloaded/compiled before the user records.
final class AppModels: ObservableObject {
    let emotion = EmotionModel()
    let yamnet = YamnetModel()

    private var preloaded = false

    /// Download + compile every model up front (each runs on its own queue).
    func preloadAll() {
        guard !preloaded else { return }
        preloaded = true
        emotion.preload()
        yamnet.preload()
    }
}

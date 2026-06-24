import Foundation
import SwiftUI

/// Owns the on-device models for the app's lifetime and preloads them at launch so
/// the NPU models are downloaded/compiled before the user starts a check-in.
final class AppModels: ObservableObject {
    let voice = EmotionModel()       // wav2vec2 speech emotion
    let face = FaceEmotionModel()    // ViT facial expression

    private var preloaded = false

    /// Download + compile every model up front (each runs on its own queue).
    func preloadAll() {
        guard !preloaded else { return }
        preloaded = true
        voice.preload()
        face.preload()
    }

    /// True once both models are downloaded/compiled and ready to run.
    var bothReady: Bool {
        !voice.status.isBusy && !face.status.isBusy
            && !isFailed(voice.status) && !isFailed(face.status)
    }

    private func isFailed(_ s: ModelStatus) -> Bool {
        if case .failed = s { return true } else { return false }
    }
}

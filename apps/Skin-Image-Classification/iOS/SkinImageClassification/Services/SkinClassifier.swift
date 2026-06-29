import UIKit
import OSLog
import ZeticMLange

/// On-device skin-lesion classifier (ViT, 7 HAM10000 classes) running on Melange.
///
/// Loads/compiles the model for the NPU on first use, then turns a photo into a
/// ranked `Classification`. All SDK calls run on a private serial queue; the public
/// API is `async` so the pipeline can `await` it without blocking the main actor.
/// `@unchecked Sendable` is sound because `model` is only ever touched on `queue`.
final class SkinClassifier: @unchecked Sendable {
    private var model: ZeticMLangeModel?
    private let queue = DispatchQueue(label: "ai.zetic.medgemma.classifier", qos: .userInitiated)
    private let log = Logger(subsystem: "ai.zetic.demo.SkinImageClassification", category: "classifier")

    var isLoaded: Bool { model != nil }

    /// Download (first run) + compile the model. `onProgress` reports 0…1 during the
    /// actual file download; idempotent.
    func ensureLoaded(onProgress: @escaping (Float) -> Void) async throws {
        if model != nil { return }
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            queue.async { [weak self] in
                guard let self else { cont.resume(); return }
                do {
                    let m = try ZeticMLangeModel(
                        personalKey: AppConfig.personalKey,
                        name: AppConfig.Model.classifier,
                        version: AppConfig.Model.classifierVersion,
                        modelMode: ModelMode.RUN_AUTO,
                        onDownload: { progress in onProgress(progress) }
                    )
                    self.model = m
                    MemoryProbe.log("classifier loaded")
                    cont.resume()
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Preprocess `image`, run inference, and return the ranked distribution.
    func classify(_ image: UIImage) async throws -> Classification {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Classification, Error>) in
            queue.async { [weak self] in
                guard let self else { cont.resume(throwing: SimpleError("Classifier deallocated")); return }
                guard let model = self.model else {
                    cont.resume(throwing: SimpleError("Classifier not loaded")); return
                }
                self.log.info("classify: preprocessing image…")
                guard let input = ImagePreprocessor.tensor(from: image) else {
                    cont.resume(throwing: SimpleError("Couldn't read that image")); return
                }
                self.log.info("classify: preprocess done (shape \(input.shape, privacy: .public)); running model…")
                MemoryProbe.log("classifier: before run")
                do {
                    let (outputs, ms) = try measureMs { try model.run(inputs: [input]) }
                    self.log.info("classify: model.run DONE in \(Int(ms))ms, outputs=\(outputs.count)")
                    guard let first = outputs.first else {
                        cont.resume(throwing: SimpleError("Model returned no output")); return
                    }
                    let logits = MelangeKit.floats(from: first)
                    self.log.info("classify: logits count=\(logits.count)")
                    guard logits.count >= SkinClass.allCases.count else {
                        cont.resume(throwing: SimpleError("Unexpected model output size \(logits.count)")); return
                    }
                    let result = Classification(
                        logits: Array(logits.prefix(SkinClass.allCases.count)),
                        latencyMs: ms
                    )
                    cont.resume(returning: result)
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}

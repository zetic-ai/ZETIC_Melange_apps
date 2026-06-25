import Foundation
import ZeticMLange

/// Thin wrapper over ZeticMLangeModel: loads once, runs inference, times latency.
final class RPPGModel {
    private var model: ZeticMLangeModel?
    private(set) var lastLatencyMs: Double = 0

    enum ModelError: Error { case notLoaded, emptyOutput }

    /// Loads/downloads the model on a background queue. Callbacks fire on the main thread.
    func load(onProgress: @escaping (Float) -> Void,
              completion: @escaping (Result<Void, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let m = try ZeticMLangeModel(
                    personalKey: AppConfig.personalKey,
                    name: AppConfig.modelName,
                    version: AppConfig.modelVersion,   // nil = latest
                    modelMode: .RUN_AUTO,
                    onDownload: { progress in
                        DispatchQueue.main.async { onProgress(progress) }
                    }
                )
                self.model = m
                MemoryProbe.log("model loaded")
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    /// Runs inference and returns the 180-sample rPPG waveform.
    func infer(_ input: Tensor) throws -> [Float] {
        guard let model else { throw ModelError.notLoaded }
        let t0 = CFAbsoluteTimeGetCurrent()
        let outputs = try model.run(inputs: [input])
        lastLatencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        guard let first = outputs.first else { throw ModelError.emptyOutput }
        return DataUtils.dataToFloatArray(first.data)
    }
}

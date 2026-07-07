import Foundation
import ZeticMLange

/// Thin wrapper over ZeticMLangeModel: loads once, runs depth inference, times latency.
final class DepthModel {
    private var model: ZeticMLangeModel?
    private(set) var lastInferMs: Double = 0

    enum ModelError: Error, LocalizedError {
        case notLoaded, emptyOutput

        var errorDescription: String? {
            switch self {
            case .notLoaded: return "Model is not loaded yet"
            case .emptyOutput: return "Model returned no output tensors"
            }
        }
    }

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
                print("[ImageTo3D] model loaded: \(AppConfig.modelName)")
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                print("[ImageTo3D] model load FAILED: \(error)")
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    /// Runs depth inference on a CHW planar [0, 1] RGB buffer.
    func infer(_ chw: [Float]) throws -> DepthMap {
        guard let model else { throw ModelError.notLoaded }
        let size = AppConfig.inputSize

        let data = chw.withUnsafeBufferPointer { Data(buffer: $0) }
        let input = ZeticMLange.Tensor(data: data,
                                       dataType: ZeticMLange.BuiltinDataType.float32,
                                       shape: [1, 3, size, size])

        let t0 = CFAbsoluteTimeGetCurrent()
        let outputs = try model.run(inputs: [input])
        lastInferMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        guard let first = outputs.first else { throw ModelError.emptyOutput }
        return try DepthMap(tensor: first)
    }
}

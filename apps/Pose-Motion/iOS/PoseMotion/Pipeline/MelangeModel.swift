import Foundation
import ZeticMLange

/// Thin wrapper over ZeticMLangeModel: loads once, runs inference, times latency.
final class MelangeModel {
    let label: String
    private var model: ZeticMLangeModel?
    private(set) var lastLatencyMs: Double = 0

    enum ModelError: Error { case notLoaded, emptyOutput }

    var isLoaded: Bool { model != nil }

    init(label: String) {
        self.label = label
    }

    /// Loads/downloads the model on a background queue. Callbacks fire on the main thread.
    /// Pass `target` to pin a specific runtime (e.g. CoreML/ANE) instead of RUN_AUTO —
    /// needed where RUN_AUTO's GPU path aborts in MPSGraph on some model/OS combos.
    func load(name: String, version: Int?,
              target: Target? = nil, apType: APType = .NA,
              onProgress: @escaping (Float) -> Void,
              completion: @escaping (Result<Void, Error>) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let progress: (Float) -> Void = { p in
                    DispatchQueue.main.async { onProgress(p) }
                }
                let m: ZeticMLangeModel
                if let target {
                    m = try ZeticMLangeModel(
                        personalKey: AppConfig.personalKey,
                        name: name,
                        version: version,      // nil = latest
                        target: target,
                        apType: apType,
                        onDownload: progress
                    )
                } else {
                    m = try ZeticMLangeModel(
                        personalKey: AppConfig.personalKey,
                        name: name,
                        version: version,
                        modelMode: .RUN_AUTO,
                        onDownload: progress
                    )
                }
                self.model = m
                MemoryProbe.log("\(self.label) loaded")
                DispatchQueue.main.async { completion(.success(())) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    private var firstRunDone = false

    func run(_ inputs: [Tensor]) throws -> [Tensor] {
        guard let model else { throw ModelError.notLoaded }
        #if DEBUG
        if !firstRunDone { print("[run] \(label) first inference…") }
        #endif
        let t0 = CFAbsoluteTimeGetCurrent()
        let outputs = try model.run(inputs: inputs)
        lastLatencyMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        #if DEBUG
        if !firstRunDone {
            firstRunDone = true
            print(String(format: "[run] %@ ok (%.1f ms)", label, lastLatencyMs))
        }
        #endif
        return outputs
    }
}

import Foundation

actor ModelPreparationManager {
    func prepare(using engine: TranslationEngine, onProgress: @Sendable @escaping (Double) -> Void) async -> Bool {
        await MainActor.run { onProgress(0.05) }
        // Simulate staged progress while model loads
        for step in 1...8 {
            try? await Task.sleep(nanoseconds: 80_000_000)
            let p = 0.05 + (Double(step) * 0.1)
            await MainActor.run { onProgress(min(p, 0.85)) }
        }
        let ok = await engine.warmup()
        await MainActor.run { onProgress(1.0) }
        return ok
    }
}

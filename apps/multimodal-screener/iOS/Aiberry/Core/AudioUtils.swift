import Foundation

/// Lightweight waveform helpers to feed the emotion model cleaner input and to
/// derive a simple "rate of speech" proxy for the fusion engine.
enum AudioUtils {

    /// Energy-based trim of leading/trailing silence (simple VAD).
    /// Keeps frames whose RMS is ≥ `relThreshold` × the loudest frame, plus a margin.
    static func trimSilence(_ x: [Float],
                            frame: Int = 320,            // 20 ms @ 16 kHz
                            relThreshold: Float = 0.08) -> [Float] {
        guard x.count > frame else { return x }

        let energies = frameEnergies(x, frame: frame)
        guard let maxE = energies.max(), maxE > 1e-6 else { return x }
        let thr = maxE * relThreshold
        guard let firstVoiced = energies.firstIndex(where: { $0 >= thr }),
              let lastVoiced = energies.lastIndex(where: { $0 >= thr }) else { return x }

        let start = max(0, (firstVoiced - 1) * frame)            // one-frame margin
        let stop  = min(x.count, (lastVoiced + 2) * frame)
        guard stop > start else { return x }
        return Array(x[start..<stop])
    }

    /// Split into up to `maxWindows` windows of `size` samples for multi-window
    /// averaging. Short input returns a single window (caller pads/tiles it).
    static func windows(_ x: [Float], size: Int, maxWindows: Int = 3) -> [[Float]] {
        guard x.count > size else { return [x] }
        let count = min(maxWindows, Int((Double(x.count) / Double(size)).rounded(.up)))
        if count <= 1 { return [Array(x.prefix(size))] }
        let stride = (x.count - size) / (count - 1)
        return (0..<count).map { Array(x[$0 * stride ..< $0 * stride + size]) }
    }

    /// Fraction of the clip that is voiced (frame RMS ≥ `relThreshold` × loudest).
    /// A transparent "rate of speech" proxy: more time actually speaking → higher value.
    /// Returns 0…1.
    static func voicedFraction(_ x: [Float],
                               frame: Int = 320,
                               relThreshold: Float = 0.08) -> Float {
        guard x.count >= frame else { return 0 }
        let energies = frameEnergies(x, frame: frame)
        guard let maxE = energies.max(), maxE > 1e-6 else { return 0 }
        let thr = maxE * relThreshold
        let voiced = energies.reduce(into: 0) { $0 += ($1 >= thr ? 1 : 0) }
        return Float(voiced) / Float(energies.count)
    }

    /// Per-frame RMS energies over non-overlapping `frame`-sample windows.
    private static func frameEnergies(_ x: [Float], frame: Int) -> [Float] {
        var energies: [Float] = []
        energies.reserveCapacity(x.count / frame + 1)
        var i = 0
        while i < x.count {
            let end = min(i + frame, x.count)
            var sum: Float = 0
            for j in i..<end { sum += x[j] * x[j] }
            energies.append((sum / Float(end - i)).squareRoot())
            i += frame
        }
        return energies
    }
}

import Foundation

/// Lightweight waveform helpers to feed the emotion model cleaner input.
///
/// Two issues hurt accuracy with real recordings (vs the model's clean, all-speech
/// training clips): (1) dead air before/after speech dilutes the model's mean-pool,
/// and (2) speech longer than the 3 s window gets cropped. These helpers trim
/// silence and split long speech into overlapping windows for logit averaging.
enum AudioUtils {

    /// Energy-based trim of leading/trailing silence (simple VAD).
    /// Keeps frames whose RMS is ≥ `relThreshold` × the loudest frame, plus a margin.
    static func trimSilence(_ x: [Float],
                            frame: Int = 320,            // 20 ms @ 16 kHz
                            relThreshold: Float = 0.08) -> [Float] {
        guard x.count > frame else { return x }

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
}

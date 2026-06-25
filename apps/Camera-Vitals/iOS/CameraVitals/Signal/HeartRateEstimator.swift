import Foundation

/// Turns a model rPPG waveform (DiffNormalized derivative) into a heart rate.
/// Mirrors rPPG-Toolbox postprocess: cumsum → detrend → bandpass → FFT peak.
enum HeartRateEstimator {
    struct Result {
        let bpm: Double
        let quality: Double      // 0...1
        let filtered: [Float]
    }

    static func estimate(_ waveform: [Float], fs: Double) -> Result? {
        guard waveform.count >= 60 else { return nil }

        // Integrate (model predicts the derivative of the pulse).
        var cumulative = [Float](repeating: 0, count: waveform.count)
        var acc: Float = 0
        for i in 0..<waveform.count { acc += waveform[i]; cumulative[i] = acc }

        let detrended = Detrend.linear(cumulative)
        let filtered = BandpassHR.filtfilt(detrended)

        guard let peak = FFTAnalyzer.dominantBPM(
            filtered, fs: fs, lo: AppConfig.hrBandLow, hi: AppConfig.hrBandHigh
        ) else { return nil }

        guard peak.bpm >= AppConfig.minBPM, peak.bpm <= AppConfig.maxBPM else {
            return Result(bpm: peak.bpm, quality: 0, filtered: filtered)  // physiologically implausible
        }

        // Map harmonic SNR (dB) → 0...1 quality. Calibrated against scripts: a real pulse
        // (≳ +3 dB) reaches green (≥0.6), a usable pulse amber (≥0.3 ≈ −0.4 dB), pure noise
        // (≈ −1 dB) stays red.
        let quality = ((peak.snr + 4.0) / 12.0).clamped(to: 0...1)
        return Result(bpm: peak.bpm, quality: quality, filtered: filtered)
    }
}

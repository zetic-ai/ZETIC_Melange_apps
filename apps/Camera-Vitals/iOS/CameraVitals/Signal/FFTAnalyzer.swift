import Accelerate
import Foundation

struct SpectrumPeak {
    let bpm: Double
    let snr: Double      // dB: peak power vs rest of the HR band
}

/// Finds the dominant frequency of a 1-D signal inside the heart-rate band
/// using a Hann-windowed, zero-padded FFT with quadratic peak interpolation.
enum FFTAnalyzer {
    static func dominantBPM(_ signal: [Float], fs: Double, lo: Double, hi: Double) -> SpectrumPeak? {
        let n = signal.count
        guard n >= 16 else { return nil }

        let log2n = vDSP_Length(ceil(log2(Double(n))))
        let fftN = Int(1) << Int(log2n)
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        defer { vDSP_destroy_fftsetup(setup) }

        // Hann window over the populated samples.
        var hann = [Float](repeating: 0, count: n)
        vDSP_hann_window(&hann, vDSP_Length(n), Int32(vDSP_HANN_NORM))
        var windowed = [Float](repeating: 0, count: n)
        vDSP_vmul(signal, 1, hann, 1, &windowed, 1, vDSP_Length(n))

        var real = [Float](repeating: 0, count: fftN)
        var imag = [Float](repeating: 0, count: fftN)
        for i in 0..<n { real[i] = windowed[i] }

        let half = fftN / 2
        var mag = [Float](repeating: 0, count: half)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                vDSP_fft_zip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                vDSP_zvabs(&split, 1, &mag, 1, vDSP_Length(half))
            }
        }

        let loBin = Int(ceil(lo * Double(fftN) / fs))
        let hiBin = Int(floor(hi * Double(fftN) / fs))
        guard loBin >= 1, hiBin < half, hiBin > loBin else { return nil }

        var peakBin = loBin
        var peakVal = mag[loBin]
        var bandPower: Float = 0
        for k in loBin...hiBin {
            bandPower += mag[k] * mag[k]
            if mag[k] > peakVal { peakVal = mag[k]; peakBin = k }
        }

        // Quadratic interpolation around the peak bin for sub-bin resolution.
        var interp = Double(peakBin)
        if peakBin > loBin, peakBin < hiBin {
            let a = Double(mag[peakBin - 1]), b = Double(mag[peakBin]), c = Double(mag[peakBin + 1])
            let denom = a - 2 * b + c
            if abs(denom) > 1e-9 { interp += 0.5 * (a - c) / denom }
        }

        let f0 = fs * interp / Double(fftN)
        let bpm = f0 * 60.0

        // SNR = energy concentrated at the pulse fundamental + 2nd harmonic vs the rest of
        // the band (rPPG-toolbox style). A clean pulse concentrates energy → positive dB.
        let halfWin = 0.15   // Hz (~9 bpm) window around each harmonic
        var signalPower: Float = 0
        for k in loBin...hiBin {
            let f = fs * Double(k) / Double(fftN)
            if abs(f - f0) <= halfWin || abs(f - 2 * f0) <= halfWin {
                signalPower += mag[k] * mag[k]
            }
        }
        let noisePower = max(bandPower - signalPower, 1e-9)
        let snr = signalPower > 0 ? 10 * log10(Double(signalPower) / Double(noisePower)) : -20
        return SpectrumPeak(bpm: bpm, snr: snr)
    }
}

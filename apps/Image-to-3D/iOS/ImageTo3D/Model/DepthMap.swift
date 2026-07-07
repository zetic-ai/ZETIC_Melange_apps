import Foundation
import ZeticMLange

/// Depth map decoded from the model's output tensor.
/// Values are relative inverse depth: larger = closer to the camera.
struct DepthMap {
    let values: [Float]
    let width: Int
    let height: Int
    /// Robust range (2nd / 98th percentile) — raw min/max lets a few outlier
    /// pixels compress the whole usable range.
    let robustLo: Float
    let robustHi: Float

    enum ParseError: Error, LocalizedError {
        case sizeMismatch(String)

        var errorDescription: String? {
            if case .sizeMismatch(let detail) = self { return detail }
            return nil
        }
    }

    /// Parses the output tensor defensively: expected shape is [1, 518, 518],
    /// but tolerate [1, 1, H, W] / [H, W] / a flat buffer by taking the last
    /// two dims as H×W (square-root fallback when no shape is reported).
    init(tensor: ZeticMLange.Tensor) throws {
        let floats: [Float] = tensor.data.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }

        let dims = tensor.shape.filter { $0 > 1 }
        var h: Int
        var w: Int
        if dims.count >= 2 {
            h = dims[dims.count - 2]
            w = dims[dims.count - 1]
        } else {
            let side = Int(Double(floats.count).squareRoot().rounded())
            h = side
            w = side
        }
        guard h * w == floats.count else {
            throw ParseError.sizeMismatch(
                "output shape \(tensor.shape) (\(h)x\(w)) does not match \(floats.count) floats")
        }

        // Percentiles from a 1/16 subsample — plenty for a stable estimate.
        var sample = [Float]()
        sample.reserveCapacity(floats.count / 16 + 1)
        for i in stride(from: 0, to: floats.count, by: 16) {
            sample.append(floats[i])
        }
        sample.sort()
        let lo = sample[Int(Float(sample.count - 1) * 0.02)]
        let hi = sample[Int(Float(sample.count - 1) * 0.98)]

        self.values = floats
        self.width = w
        self.height = h
        self.robustLo = lo
        self.robustHi = max(hi, lo + 1e-6)
    }

    /// Disparity normalized to [0, 1] over the robust range; 1 = closest.
    func normalized(_ index: Int) -> Float {
        min(max((values[index] - robustLo) / (robustHi - robustLo), 0), 1)
    }
}

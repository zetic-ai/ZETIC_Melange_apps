import Foundation
import ZeticMLange

/// Thin helpers around the ZeticMLange 1.6.0 Swift API so each model file stays small.
///
/// API recap (from the SDK's swiftinterface):
///   `ZeticMLangeModel(personalKey:name:version:modelMode:onDownload:)`
///   `func run(inputs: [Tensor]) throws -> [Tensor]`
///   `Tensor(data: Data, dataType: any DataType, shape: [Int])`
enum MelangeKit {

    /// Load (and on first run, download + compile) a Melange model for this device's NPU.
    static func load(_ name: String, version: Int = 1,
                     onProgress: @escaping (Float) -> Void) throws -> ZeticMLangeModel {
        try ZeticMLangeModel(
            personalKey: AppConfig.personalKey,
            name: name,
            version: version,
            modelMode: ModelMode.RUN_ACCURACY,
            onDownload: { progress in onProgress(progress) }
        )
    }

    /// Wrap a Float array as a float32 input tensor with the given shape.
    static func floatTensor(_ values: [Float], shape: [Int]) -> Tensor {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.float32, shape: shape)
    }

    /// Read raw tensor bytes back into a Float array.
    static func floats(from data: Data) -> [Float] {
        let count = data.count / MemoryLayout<Float>.size
        guard count > 0 else { return [] }
        return data.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: Float.self).baseAddress!
            return Array(UnsafeBufferPointer(start: base, count: count))
        }
    }

    static func floats(from tensor: Tensor) -> [Float] { floats(from: tensor.data) }

    /// Pad with zeros / trim (from the start) so a clip is exactly `count` samples,
    /// to match a model's fixed NPU input length.
    static func fit(_ samples: [Float], to count: Int) -> [Float] {
        if samples.count == count { return samples }
        if samples.count > count { return Array(samples.prefix(count)) }
        return samples + [Float](repeating: 0, count: count - samples.count)
    }

    /// Fit to exactly `count` samples by **tiling** short clips (repeating the speech)
    /// instead of zero-padding, and center-cropping long ones. Repeating avoids diluting
    /// a mean-pooled model with silence frames — better for short emotion clips.
    static func fitTiling(_ samples: [Float], to count: Int) -> [Float] {
        guard !samples.isEmpty else { return [Float](repeating: 0, count: count) }
        if samples.count == count { return samples }
        if samples.count > count {
            let start = (samples.count - count) / 2          // centered crop
            return Array(samples[start..<start + count])
        }
        var out = [Float]()
        out.reserveCapacity(count)
        while out.count < count { out.append(contentsOf: samples) }
        return Array(out.prefix(count))
    }

    /// Numerically stable softmax.
    static func softmax(_ x: [Float]) -> [Float] {
        guard let m = x.max() else { return x }
        let e = x.map { Foundation.exp($0 - m) }
        let sum = e.reduce(0, +)
        return sum > 0 ? e.map { $0 / sum } : e
    }
}

/// Measures wall-clock duration of a throwing block, in milliseconds.
@inline(__always)
func measureMs<T>(_ body: () throws -> T) rethrows -> (value: T, ms: Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let value = try body()
    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    return (value, ms)
}

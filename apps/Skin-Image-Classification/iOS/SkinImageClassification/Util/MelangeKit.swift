import Foundation
import ZeticMLange

/// Thin helpers around the ZeticMLange 1.6.0 Swift API so each model file stays small.
///
/// API recap (from the SDK's swiftinterface):
///   `ZeticMLangeModel(personalKey:name:version:modelMode:onDownload:)`
///   `func run(inputs: [Tensor]) throws -> [Tensor]`
///   `Tensor(data: Data, dataType: any DataType, shape: [Int])`
enum MelangeKit {

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
}

/// Measures wall-clock duration of a throwing block, in milliseconds.
@inline(__always)
func measureMs<T>(_ body: () throws -> T) rethrows -> (value: T, ms: Double) {
    let start = DispatchTime.now().uptimeNanoseconds
    let value = try body()
    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    return (value, ms)
}

struct SimpleError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}

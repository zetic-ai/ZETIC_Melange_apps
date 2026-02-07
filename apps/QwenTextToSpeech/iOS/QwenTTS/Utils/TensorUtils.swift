import Foundation
import ZeticMLange

// Helper to create Tensor from [Float] (Float32)
func makeFloatTensor(values: [Float], shape: [Int]) -> Tensor {
    let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
    return Tensor(data: data, dataType: BuiltinDataType.float32, shape: shape)
}

// Helper to create Tensor from [Float] converting to Float16
func makeFloat16Tensor(values: [Float], shape: [Int]) -> Tensor {
    // Float16 available in iOS 14+
    let float16Values = values.map { Float16($0) }
    let data = float16Values.withUnsafeBufferPointer { Data(buffer: $0) }
    return Tensor(data: data, dataType: BuiltinDataType.float16, shape: shape)
}

// Helper to decode [Float] from Tensor
func decodeFloatArray(from tensor: Tensor) -> [Float] {
    let data = tensor.data
    if let type = tensor.dataType as? BuiltinDataType, type == .float16 {
        return data.withUnsafeBytes { pointer in
            let buffer = pointer.bindMemory(to: Float16.self)
            return buffer.map { Float($0) }
        }
    } else {
        let count = data.count / MemoryLayout<Float>.size
        return data.withUnsafeBytes { pointer in
            let buffer = pointer.bindMemory(to: Float.self)
            return Array(buffer.prefix(count))
        }
    }
}

// Helper: Argmax over last dimension (flattened input)
// Input: [N * C * D], Shape [N, C, D]
// Output: [N * C] indices as Float
func argmax(logits: [Float], groupSize: Int) -> [Float] {
    let numGroups = logits.count / groupSize
    var indices: [Float] = []
    indices.reserveCapacity(numGroups)
    
    for i in 0..<numGroups {
        let start = i * groupSize
        let end = start + groupSize
        // Find max in slice
        var maxVal = -Float.greatestFiniteMagnitude
        var maxIdx = 0
        for j in start..<end {
            if logits[j] > maxVal {
                maxVal = logits[j]
                maxIdx = j - start
            }
        }
        indices.append(Float(maxIdx))
    }
    return indices
}

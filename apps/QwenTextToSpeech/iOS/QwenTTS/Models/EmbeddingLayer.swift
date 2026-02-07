import Foundation

class EmbeddingLayer {
    private var data: Data?
    private let embeddingDim: Int
    private let vocabSize: Int
    
    init(fileName: String, vocabSize: Int, embeddingDim: Int) {
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.load(fileName: fileName)
    }
    
    private func load(fileName: String) {
        // Look for the file in the bundle
        // Note: For large files, we might want to use memory mapping (NSData.ReadingOptions.mappedIfSafe)
        if let path = Bundle.main.path(forResource: fileName, ofType: "bin") {
            do {
                self.data = try Data(contentsOf: URL(fileURLWithPath: path), options: .mappedIfSafe)
                print("Loaded embedding \(fileName) of size \(self.data?.count ?? 0)")
            } catch {
                print("Failed to load embedding file \(fileName): \(error)")
            }
        } else {
             // Fallback: Check if file exists in document directory or just try generic path (for testing)
             print("Embedding file \(fileName).bin not found in Bundle")
        }
    }
    
    func get(indices: [Int]) -> [Float] {
        guard let data = data else { return [] }
        
        var result: [Float] = []
        result.reserveCapacity(indices.count * embeddingDim)
        
        let floatSize = MemoryLayout<Float>.size
        let totalFloats = data.count / floatSize
        
        data.withUnsafeBytes { buffer in
            guard let baseAddress = buffer.baseAddress?.assumingMemoryBound(to: Float.self) else { return }
            
            for index in indices {
                if index >= 0 && index < vocabSize {
                    // Calculate offset
                    let startOffset = index * embeddingDim
                    if startOffset + embeddingDim <= totalFloats {
                        let tensorPtr = baseAddress.advanced(by: startOffset)
                        // Create a buffer pointer for this slice
                        let bufferPtr = UnsafeBufferPointer(start: tensorPtr, count: embeddingDim)
                        result.append(contentsOf: bufferPtr)
                    } else {
                       // Index out of bounds of file
                       result.append(contentsOf: Array(repeating: 0.0, count: embeddingDim))
                    }
                } else {
                    // Padding or invalid index
                    result.append(contentsOf: Array(repeating: 0.0, count: embeddingDim))
                }
            }
        }
        
        return result
    }
}

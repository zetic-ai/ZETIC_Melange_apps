import Foundation
import ZeticMLange

struct TTSModelConfig {
    // Set via Xcode Environment Variables (Edit Scheme -> Arguments -> Environment Variables)
    // ZETIC_TOKEN_KEY: Your personal Zetic access token
    static var tokenKey: String {
        ProcessInfo.processInfo.environment["ZETIC_TOKEN_KEY"] ?? ""
    }
    
    struct TextProjection {
        static let name = "jathin-zetic/qwen_tts06b_text_projection"
        static let version = 1
    }
    
    struct CodePredictor {
        static let name = "jathin-zetic/qwen_tts06b_code_predictor"
        static let version = 1
    }
    
    struct Talker {
        static let name = "jathin-zetic/qwen_tts06b_talker"
        static let version = 1
    }
    
    struct CodecHead {
        static let name = "jathin-zetic/qwen_tts06b_codec_head"
        static let version = 1
    }
    
    struct SpeechDecode {
        static let name = "jathin-zetic/qwen_tts06b_speech_decode"
        static let version = 3
    }
}

/// Wrapper for ZeticMLange models
/// SDK v1.4.5 API: ZeticMLangeModel(tokenKey: String, name: String, version: Int)
///                 model.run(inputs: [Tensor]) -> [Tensor]
class ZeticModelWrapper {
    private var model: ZeticMLangeModel?
    private let modelName: String
    private let modelVersion: Int
    private let target: ZeticMLange.Target?
    
    init(name: String, version: Int, target: ZeticMLange.Target? = nil) {
        self.modelName = name
        self.modelVersion = version
        self.target = target
    }
    
    func load() async throws {
        let token = TTSModelConfig.tokenKey
        do {
            if let target = target {
                self.model = try ZeticMLangeModel(tokenKey: token, name: modelName, version: modelVersion, target: target)
            } else {
                self.model = try ZeticMLangeModel(tokenKey: token, name: modelName, version: modelVersion)
            }
        } catch {
            print("❌ FAILED: \(modelName) - \(error.localizedDescription)")
            throw error
        }
    }
    
    /// Run the model with Tensor inputs, returns Tensor outputs
    func run(inputs: [Tensor]) throws -> [Tensor] {
        guard let model = self.model else {
            let errorMsg = "Model \(modelName) not loaded"
            print("❌ [ZeticModel] \(errorMsg)")
            throw NSError(domain: "ZeticModel", code: 500, userInfo: [NSLocalizedDescriptionKey: errorMsg])
        }
        
        do {
            let outputs = try model.run(inputs: inputs)
            return outputs
        } catch {
            print("❌ Run failed: \(modelName) - \(error.localizedDescription)")
            throw error
        }
    }
    
    var isLoaded: Bool {
        return model != nil
    }
}

class TextProjectionModel: ZeticModelWrapper {
    init() {
        super.init(name: TTSModelConfig.TextProjection.name, version: TTSModelConfig.TextProjection.version)
    }
    
    func run(embedding: [Float], seqLen: Int) throws -> [Float] {
        // Shape: [1, seq_len, 2048]
        let inputTensor = makeFloatTensor(values: embedding, shape: [1, seqLen, 2048])
        let outputs = try super.run(inputs: [inputTensor])
        return outputs.first.map { decodeFloatArray(from: $0) } ?? []
    }
}

class CodePredictorModel: ZeticModelWrapper {
    init() {
        super.init(name: TTSModelConfig.CodePredictor.name, version: TTSModelConfig.CodePredictor.version)
    }
    
    func run(projectedText: [Float], validLen: Int, totalLen: Int) throws -> [Float] {
        // Input 1: Hidden States [1, totalLen, 2048]
        // Model expects Float16
        let inputTensor = makeFloat16Tensor(values: projectedText, shape: [1, totalLen, 2048])
        
        // Input 2: Attention Mask [1, 1, 1, totalLen]
        // Note: Mask expects Float32 (4 bytes/element), while features expect Float16
        var maskValues = Array(repeating: Float(0.0), count: totalLen)
        for i in 0..<validLen {
            maskValues[i] = 1.0
        }
        let maskTensor = makeFloatTensor(values: maskValues, shape: [1, 1, 1, totalLen])
        
        let outputs = try super.run(inputs: [inputTensor, maskTensor])
        return outputs.first.map { decodeFloatArray(from: $0) } ?? []
    }
}

class TalkerModel: ZeticModelWrapper {
    init() {
        super.init(name: TTSModelConfig.Talker.name, version: TTSModelConfig.Talker.version)
    }
    
    func run(codes: [Float], codesShape: [Int], speakerEmbedding: [Float]) throws -> [Float] {
        let codesTensor = makeFloatTensor(values: codes, shape: codesShape)
        let spkTensor = makeFloatTensor(values: speakerEmbedding, shape: [1, speakerEmbedding.count])
        let outputs = try super.run(inputs: [codesTensor, spkTensor])
        return outputs.first.map { decodeFloatArray(from: $0) } ?? []
    }
}

class SpeechDecodeModel: ZeticModelWrapper {
    init() {
        super.init(name: TTSModelConfig.SpeechDecode.name, version: TTSModelConfig.SpeechDecode.version, target: .ZETIC_MLANGE_TARGET_COREML)
    }
    
    // Accepts a prepared input tensor (chunk)
    func run(inputTensor: Tensor) throws -> [Float] {
        let outputs = try super.run(inputs: [inputTensor])
        return outputs.first.map { decodeFloatArray(from: $0) } ?? []
    }
}

class CodecHeadModel: ZeticModelWrapper {
    init() {
        super.init(name: TTSModelConfig.CodecHead.name, version: TTSModelConfig.CodecHead.version)
    }
    
    func run(audioFeatures: [Float], shape: [Int]) throws -> [Float] {
        let inputTensor = makeFloatTensor(values: audioFeatures, shape: shape)
        let outputs = try super.run(inputs: [inputTensor])
        return outputs.first.map { decodeFloatArray(from: $0) } ?? []
    }
}

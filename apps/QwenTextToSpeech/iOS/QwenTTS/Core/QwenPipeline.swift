import Foundation
import ZeticMLange

class QwenPipeline: ObservableObject {
    @Published var status: String = "Idle"
    @Published var isReady: Bool = false
    @Published var errorMessage: String? = nil
    @Published var loadProgress: Float = 0.0
    
    private let tokenizer = Tokenizer()
    private var textEmbedding: EmbeddingLayer?
    private var codecEmbedding: EmbeddingLayer?
    
    private let textProj = TextProjectionModel()
    private let codePred = CodePredictorModel()
    private let talker = TalkerModel()
    private let speechDec = SpeechDecodeModel()
    private let codecHead = CodecHeadModel()
    
    // Speaker Embeddings
    private var speakerMap: [String: [Float]] = [:]
    
    // Static configuration for SpeechDecode v3
    private let CHUNK_SIZE = 32
    private let NUM_CHANNELS = 16
    private let UPSAMPLE_FACTOR = 1920
    private let MAX_SEQ_LEN = 512
    private let CODE_SEQ_LEN = 128
    
    func load() async {
        print("🚀 Starting QwenTTS Pipeline Load (v3)")
        
        DispatchQueue.main.async { 
            self.status = "Loading embeddings..."
            self.errorMessage = nil
            self.loadProgress = 0.0
        }
        
        // Load embeddings
        textEmbedding = EmbeddingLayer(fileName: "text_embedding", vocabSize: 151936, embeddingDim: 2048)
        codecEmbedding = EmbeddingLayer(fileName: "codec_embedding", vocabSize: 2048, embeddingDim: 1024)
        setupSpeakers()
        
        // Load models with progress
        let models: [(String, () async throws -> Void)] = [
            ("TextProjection", { try await self.textProj.load() }),
            ("CodePredictor", { try await self.codePred.load() }),
            ("Talker", { try await self.talker.load() }),
            ("CodecHead", { try await self.codecHead.load() }),
            ("SpeechDecode", { try await self.speechDec.load() })
        ]
        
        let total = models.count
        
        do {
            for (index, (name, loadFunc)) in models.enumerated() {
                let progress = Float(index) / Float(total)
                
                print("[\(index + 1)/\(total)] Loading \(name)...")
                DispatchQueue.main.async { 
                    self.status = "Loading \(name)..."
                    self.loadProgress = progress
                }
                
                try await loadFunc()
            }
            
            print("✅ All models loaded successfully!")
            
            DispatchQueue.main.async {
                self.status = "Ready"
                self.isReady = true
                self.loadProgress = 1.0
            }
        } catch {
            print("❌ Load failed: \(error.localizedDescription)")
            
            DispatchQueue.main.async { 
                self.status = "Error: \(error.localizedDescription)"
                self.errorMessage = error.localizedDescription
            }
        }
    }
    
    func generate(text: String, speaker: String) async throws -> [Float] {
        guard isReady else { 
            throw NSError(domain: "Pipeline", code: 500, userInfo: [NSLocalizedDescriptionKey: "Not Ready"]) 
        }
        
        print("🎤 Generating for: '\(text.prefix(30))...'")
        
        // 1. Tokenization
        let tokens = tokenizer.encode(text)
        let seqLen = tokens.count
        
        guard let textEmbLayer = textEmbedding else { return [] }
        var embeddedText = textEmbLayer.get(indices: tokens)
        
        // Pad or Truncate to MAX_SEQ_LEN
        if seqLen < MAX_SEQ_LEN {
            let padLen = MAX_SEQ_LEN - seqLen
            let padSize = padLen * 2048
            embeddedText.append(contentsOf: Array(repeating: 0.0, count: padSize))
        } else if seqLen > MAX_SEQ_LEN {
            let maxSize = MAX_SEQ_LEN * 2048
            embeddedText = Array(embeddedText.prefix(maxSize))
        }
        
        DispatchQueue.main.async { self.status = "Processing Text..." }
        
        // 2. Text Projection
        let projText = try textProj.run(embedding: embeddedText, seqLen: MAX_SEQ_LEN)
        
        // 3. Code Prediction
        // Slice for CodePredictor (expects 128 tokens)
        let sliceCount = CODE_SEQ_LEN * 2048
        let slicedProjText = Array(projText.prefix(sliceCount))
        
        let codes = try codePred.run(projectedText: slicedProjText, validLen: min(seqLen, CODE_SEQ_LEN), totalLen: CODE_SEQ_LEN)
        
        // 4. Talker
        // Pad codes back to 512 tokens for Talker
        var paddedCodes = codes
        if codes.count < MAX_SEQ_LEN * 1024 {
            let paddingCount = (MAX_SEQ_LEN * 1024) - codes.count
            paddedCodes.append(contentsOf: Array(repeating: Float(0.0), count: paddingCount))
        }
        let codesShape = [1, MAX_SEQ_LEN, 1024]
        
        let spkEmb = speakerMap[speaker] ?? speakerMap.values.first ?? Array(repeating: 0.0, count: 512) 
        
        DispatchQueue.main.async { self.status = "Generating Acoustics..." }
        let acousticTokensHidden = try talker.run(codes: paddedCodes, codesShape: codesShape, speakerEmbedding: spkEmb)
        
        // 4.5 Codec Head
        // Slice back to 128 tokens for CodecHead
        let validTokenCount = CODE_SEQ_LEN * 1024
        let slicedAcousticHidden = Array(acousticTokensHidden.prefix(validTokenCount))
        
        let acousticTokensLogits = try codecHead.run(audioFeatures: slicedAcousticHidden, shape: [1, CODE_SEQ_LEN, 1024])
        
        // Convert Logits to Indices (Argmax)
        let codebookSize = 3072 / NUM_CHANNELS // 192
        let acousticTokens = argmax(logits: acousticTokensLogits, groupSize: codebookSize)
        
        // Safety Clamp to 127
        let clampedTokens = acousticTokens.map { min($0, 127) }
        
        // 5. Speech Decode (Chunked Processing)
        DispatchQueue.main.async { self.status = "Synthesizing Audio..." }
        
        var fullAudio: [Float] = []
        let totalElements = clampedTokens.count
        let totalFrames = totalElements / NUM_CHANNELS
        
        // Loop by chunks of 32 frames
        for i in stride(from: 0, to: totalFrames, by: CHUNK_SIZE) {
            let startFrame = i
            let endFrame = min(i + CHUNK_SIZE, totalFrames)
            let currentChunkSize = endFrame - startFrame
            
            // Extract chunk
            let offset = startFrame * NUM_CHANNELS
            let length = currentChunkSize * NUM_CHANNELS
            var chunkData = Array(clampedTokens[offset..<(offset + length)])
            
            // Pad last chunk if needed
            if currentChunkSize < CHUNK_SIZE {
                let padFrames = CHUNK_SIZE - currentChunkSize
                let padCount = padFrames * NUM_CHANNELS
                chunkData.append(contentsOf: Array(repeating: 0.0, count: padCount))
            }
            
            // Create Input Tensor: Transpose [Time, Channels] -> [Channels, Time]
            var transposedIndices = [Float](repeating: 0.0, count: CHUNK_SIZE * NUM_CHANNELS)
            for t in 0..<currentChunkSize {
                for c in 0..<NUM_CHANNELS {
                    let originalIdx = t * NUM_CHANNELS + c
                    let val = chunkData[originalIdx]
                    
                    let targetIdx = c * CHUNK_SIZE + t
                    transposedIndices[targetIdx] = val
                }
            }
            
            let inputTensor = makeFloatTensor(values: transposedIndices, shape: [1, NUM_CHANNELS, CHUNK_SIZE])
            
            // Run Model
            let audioChunk = try speechDec.run(inputTensor: inputTensor)
            
            // Process Output (Crop & Crossfade)
             let validSamples = currentChunkSize * UPSAMPLE_FACTOR
             let chunkToAppend = currentChunkSize < CHUNK_SIZE ? Array(audioChunk.prefix(validSamples)) : audioChunk

             if !fullAudio.isEmpty {
                 let crossfadeLen = 256
                 if fullAudio.count >= crossfadeLen && chunkToAppend.count >= crossfadeLen {
                     let startIdx = fullAudio.count - crossfadeLen
                     for j in 0..<crossfadeLen {
                         let fade = Float(j) / Float(crossfadeLen)
                         fullAudio[startIdx + j] *= (1.0 - fade)
                         fullAudio[startIdx + j] += chunkToAppend[j] * fade
                     }
                 }
             }
             fullAudio.append(contentsOf: chunkToAppend)
        }
        
        print("✅ Generation Complete: \(fullAudio.count) samples")
        DispatchQueue.main.async { self.status = "Done" }
        return fullAudio
    }
    
    private func setupSpeakers() {
        let speakers = ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"]
        for s in speakers {
            speakerMap[s] = (0..<512).map { _ in Float.random(in: -0.1...0.1) }
        }
    }
}

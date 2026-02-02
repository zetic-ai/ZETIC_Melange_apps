//
//  AnonymizerService.swift
//  TextAnonymizer
//
//  Service to handle text anonymization using Zetic MLange SDK
//

import Foundation
import ZeticMLange
import Combine

class AnonymizerViewModel: ObservableObject {
    @Published var anonymizedText = ""
    @Published var isProcessing = false
    @Published var showingError = false
    @Published var errorMessage = ""
    @Published var isModelLoaded = false
    @Published var downloadProgress: Int = 0
    
    private var model: ZeticMLangeModel?
    private var tokenizer: Tokenizer?
    private let modelMaxLength = 128
    
    // Dynamic labels
    private var id2label: [Int: String] = [:]
    
    private let placeholderByLabel: [String: String] = [
        "EMAIL": "[Email]",
        "PHONE_NUMBER": "[Phone number]",
        "CREDIT_CARD_NUMBER": "[Credit card]",
        "SSN": "[SSN]",
        "NRP": "[NRP]",
        "PERSON": "[Person]",
        "ADDRESS": "[Address]",
        "LOCATION": "[Location]",
        "DATE": "[Date]",
        "OTHER": "[Sensitive]"
    ]
    
    private let modelId: String
    private let modelVersion: Int
    private let personalKey: String

    init() {
        // Updated defaults as per user request
        let env = ProcessInfo.processInfo.environment
        modelId = env["ZETIC_MODEL_ID"] ?? "Steve/text-anonymizer-v1"
        modelVersion = Int(env["ZETIC_MODEL_VERSION"] ?? "") ?? 1
        personalKey = env["ZETIC_PERSONAL_KEY"] ?? "YOUR_MLANGE_KEY"

        loadModelAsync()
    }
    
    private func loadModelAsync() {
        print("🔄 Starting model loading...")
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                // 1. Load Tokenizer
                self.tokenizer = Tokenizer()
                
                // 2. Load Labels
                if let url = Bundle.main.url(forResource: "labels", withExtension: "json"),
                   let data = try? Data(contentsOf: url),
                   let json = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: String] {
                    var map: [Int: String] = [:]
                    for (k, v) in json {
                        if let keyInt = Int(k) {
                            map[keyInt] = v
                        }
                    }
                    self.id2label = map
                    print("Tokenizer: Loaded \(map.count) labels")
                } else {
                    print("Tokenizer: Failed to load labels.json")
                }
                
                print("📦 Attempting to load Zetic MLange model...")
                guard !personalKey.isEmpty else {
                    throw NSError(
                        domain: "TextAnonymizer",
                        code: 1001,
                        userInfo: [NSLocalizedDescriptionKey: "Missing personal key. Set ZETIC_PERSONAL_KEY in your scheme environment."]
                    )
                }

                print("   Model: \(modelId)")
                
                let loadedModel = try ZeticMLangeModel(
                    tokenKey: personalKey,
                    name: modelId,
                    version: modelVersion,
                    onDownload: { progress in
                        DispatchQueue.main.async {
                            self.downloadProgress = Int(progress * 100)
                        }
                    }
                )
                
                print("✅ Model instance created successfully")
                
                DispatchQueue.main.async {
                    self.model = loadedModel
                    self.isModelLoaded = true
                    print("✅ Model loaded and ready to use")
                }
            } catch {
                let errorDescription = error.localizedDescription
                print("❌ Model loading failed: \(errorDescription)")
                
                DispatchQueue.main.async {
                    self.isModelLoaded = false
                    self.errorMessage = "Failed to load model: \(errorDescription)"
                    self.showingError = true
                }
            }
        }
    }
    
    func anonymizeText(_ text: String) {
        guard !text.isEmpty else { return }
        guard let model = model, let tokenizer = tokenizer else {
            errorMessage = "Model or Tokenizer not loaded."
            showingError = true
            return
        }
        
        isProcessing = true
        anonymizedText = ""
        
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            
            do {
                // 1. Tokenize
                let (inputIds, attentionMask) = self.tokenize(text, tokenizer: tokenizer)
                
                // 2. Prepare Tensors
                // Pad/Truncate
                var finalInputIds = inputIds
                var finalMask = attentionMask
                
                if finalInputIds.count > modelMaxLength {
                    finalInputIds = Array(finalInputIds.prefix(modelMaxLength))
                    finalMask = Array(finalMask.prefix(modelMaxLength))
                } else {
                    let padCount = modelMaxLength - finalInputIds.count
                    finalInputIds += Array(repeating: tokenizer.padId, count: padCount)
                    finalMask += Array(repeating: 0, count: padCount)
                }
                
                let inputIdsTensor = try self.createLongTensor(from: finalInputIds, shape: [1, modelMaxLength], label: "input_ids")
                let maskTensor = try self.createLongTensor(from: finalMask, shape: [1, modelMaxLength], label: "attention_mask")
                
                // 3. Inference
                let outputs = try model.run(inputs: [inputIdsTensor, maskTensor])
                
                // 4. Post-process
                if let output = outputs.first {
                    let result = self.decodeAndAnonymize(output, inputIds: finalInputIds, attentionMask: finalMask, tokenizer: tokenizer)
                    
                    DispatchQueue.main.async {
                        self.anonymizedText = result
                        self.isProcessing = false
                    }
                } else {
                    throw NSError(domain: "TextAnonymizer", code: 2, userInfo: [NSLocalizedDescriptionKey: "No output"])
                }
                
            } catch {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.errorMessage = "Error: \(error.localizedDescription)"
                    self.showingError = true
                }
            }
        }
    }
    
    private func tokenize(_ text: String, tokenizer: Tokenizer) -> ([Int], [Int]) {
        let ids = tokenizer.encode(text)
        let mask = Array(repeating: 1, count: ids.count)
        return (ids, mask)
    }
    
    private func createLongTensor(from values: [Int], shape: [Int], label: String) throws -> Tensor {
        let int64Values = values.map { Int64($0) }
        let data = int64Values.withUnsafeBufferPointer { Data(buffer: $0) }
        return Tensor(data: data, dataType: BuiltinDataType.int64, shape: shape)
    }
    
    private func decodeAndAnonymize(_ logitsTensor: Tensor, inputIds: [Int], attentionMask: [Int], tokenizer: Tokenizer) -> String {
        // logits: [1, seqLen, classCount]
        let classCount = id2label.count
        guard classCount > 0 else { return "Labels not loaded" }
        
        // Convert data to floats
        // Assuming Float32 output
        let floatCount = logitsTensor.data.count / MemoryLayout<Float32>.size
        let floats = logitsTensor.data.withUnsafeBytes {
            Array($0.bindMemory(to: Float32.self))
        }
        
        let seqLen = floatCount / classCount
        
        // 1. Argmax
        var predIds: [Int] = []
        for i in 0..<seqLen {
            var maxScore: Float32 = -Float32.infinity
            var maxIdx = 0
            let offset = i * classCount
            
            for c in 0..<classCount {
                if offset + c < floats.count {
                    let score = floats[offset + c]
                    if score > maxScore {
                        maxScore = score
                        maxIdx = c
                    }
                }
            }
            predIds.append(maxIdx)
        }
        
        // 2. Decode
        var maskedTokens: [String] = []
        
        var i = 0
        let realLen = min(seqLen, inputIds.count)
        
        while i < realLen {
            // Padding check
            if i < attentionMask.count && attentionMask[i] == 0 {
                i += 1
                continue
            }
            
            // Special tokens check
            let currentId = inputIds[i]
            if currentId == tokenizer.bosId || currentId == tokenizer.eosId || currentId == tokenizer.padId {
                i += 1
                continue
            }
            
            let label = id2label[predIds[i]] ?? "O"
            let rawToken = tokenizer.getRawToken(currentId) ?? ""
            
            if label == "O" {
                maskedTokens.append(tokenizer.decodeToken(currentId))
                i += 1
                continue
            }
            
            // B- or I-
            var entityType = label
            if label.hasPrefix("B-") || label.hasPrefix("I-") {
                entityType = String(label.dropFirst(2))
            }
            
            var placeholder = placeholderByLabel[entityType] ?? "[\(entityType)]"
            
            // Preserve leading space if any
            if rawToken.hasPrefix("\u{0120}") {
                placeholder = "\u{0120}" + placeholder
            }
            
            maskedTokens.append(placeholder)
            
            i += 1
            // Skip consecutive same-entity
            while i < realLen {
                if i < attentionMask.count && attentionMask[i] == 0 { break }
                let nextId = inputIds[i]
                if nextId == tokenizer.eosId || nextId == tokenizer.padId { break }
                
                let nextLabel = id2label[predIds[i]] ?? "O"
                if nextLabel == "I-\(entityType)" || nextLabel == "B-\(entityType)" {
                    i += 1
                } else {
                    break
                }
            }
        }
        
        // 3. Join
        let result = maskedTokens.joined(separator: "")
            .replacingOccurrences(of: "\u{0120}", with: " ")
            
        // Reduce multiple spaces
        let pattern = "\\s+"
        if let regex = try? NSRegularExpression(pattern: pattern, options: []) {
            let range = NSRange(location: 0, length: result.utf16.count)
            return regex.stringByReplacingMatches(in: result, options: [], range: range, withTemplate: " ")
        }
        
        return result
    }
}

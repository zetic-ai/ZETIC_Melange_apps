//
//  ZeticTensorFactory.swift
//  PromptGuard
//
//  Builds [Tensor] for ZeticMLangeModel.run(inputs:) from prompt text.
//  Uses the same tokenizer as the model (Llama Prompt Guard) so results match Python:
//  tokenizer(text, return_tensors="pt") → input_ids, attention_mask [1, 128] Int32.
//

import Foundation
import ZeticMLange

enum ZeticTensorFactoryError: Error {
    case encodingFailed
}

/// When tokenizer is missing, we use byte-level (UTF-8) encoding so the app still runs; add tokenizer.json for accurate results.
struct TensorInputResult {
    let tensors: [Tensor]
    let usedTokenizer: Bool
    let promptTokenCount: Int
    let firstTokenIdsForLogging: [Int32]
}

final class ZeticTensorFactory {

    private static let seqLen = 128

    /// Build input_ids and attention_mask [1, 128] Int32. Uses tokenizer when loaded; otherwise fallback to UTF-8 bytes.
    static func createInput(prompt: String, maxTokens: Int, tokenizer: PromptGuardTokenizer?) throws -> TensorInputResult {
        if let tokenizer = tokenizer, tokenizer.isLoaded {
            let ids = tokenizer.encode(prompt)
            let padId = Int32(truncatingIfNeeded: tokenizer.padId)
            var tokenIds: [Int32]
            var promptLength: Int
            if ids.count > seqLen {
                tokenIds = ids.prefix(seqLen).map { Int32(truncatingIfNeeded: $0) }
                promptLength = seqLen
            } else {
                tokenIds = ids.map { Int32(truncatingIfNeeded: $0) }
                promptLength = tokenIds.count
                tokenIds.append(contentsOf: [Int32](repeating: padId, count: seqLen - tokenIds.count))
            }
            let tokenData = tokenIds.withUnsafeBufferPointer { Data(buffer: $0) }
            let inputIdsTensor = Tensor(data: tokenData, dataType: BuiltinDataType.int32, shape: [1, seqLen])
            var mask = [Int32](repeating: 0, count: seqLen)
            for i in 0..<promptLength { mask[i] = 1 }
            let maskData = mask.withUnsafeBufferPointer { Data(buffer: $0) }
            let maskTensor = Tensor(data: maskData, dataType: BuiltinDataType.int32, shape: [1, seqLen])
            return TensorInputResult(tensors: [inputIdsTensor, maskTensor], usedTokenizer: true, promptTokenCount: promptLength, firstTokenIdsForLogging: Array(tokenIds.prefix(12)))
        }

        // Fallback: UTF-8 byte encoding so app works without tokenizer.json
        let utf8 = Array(prompt.utf8)
        var tokenIds = utf8.map { Int32(truncatingIfNeeded: $0) }
        let promptLength = min(tokenIds.count, seqLen)
        if tokenIds.count > seqLen {
            tokenIds = Array(tokenIds.prefix(seqLen))
        } else {
            tokenIds.append(contentsOf: [Int32](repeating: 0, count: seqLen - tokenIds.count))
        }
        let tokenData = tokenIds.withUnsafeBufferPointer { Data(buffer: $0) }
        let inputIdsTensor = Tensor(data: tokenData, dataType: BuiltinDataType.int32, shape: [1, seqLen])
        var mask = [Int32](repeating: 0, count: seqLen)
        for i in 0..<promptLength { mask[i] = 1 }
        let maskData = mask.withUnsafeBufferPointer { Data(buffer: $0) }
        let maskTensor = Tensor(data: maskData, dataType: BuiltinDataType.int32, shape: [1, seqLen])
        return TensorInputResult(tensors: [inputIdsTensor, maskTensor], usedTokenizer: false, promptTokenCount: promptLength, firstTokenIdsForLogging: Array(tokenIds.prefix(12)))
    }
}

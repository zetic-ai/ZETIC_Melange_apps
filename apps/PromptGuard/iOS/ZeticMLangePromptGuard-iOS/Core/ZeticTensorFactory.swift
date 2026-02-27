//
//  ZeticTensorFactory.swift
//  PromptGuard
//
//  Centralized adapter: builds [Tensor] for ZeticMLangeModel.run(inputs:) from prompt text.
//  Input format: single tensor of Int32 token IDs, shape [1, maxTokens].
//

import Foundation
import ZeticMLange

enum ZeticTensorFactoryError: Error {
    case encodingFailed
}

/// Creates model input [Tensor] from prompt string per ModelInputSpec.
/// Uses UTF-8 byte-as-token encoding (each byte 0–255 → Int32) and pads to maxTokens.
final class ZeticTensorFactory {

    /// Produces [Tensor] suitable for ZeticMLangeModel.run(inputs:).
    /// - Parameter prompt: Full prompt string (after applying template).
    /// - Parameter maxTokens: Sequence length (pad/truncate to this).
    /// - Returns: Array with one Tensor (Int32, shape [1, maxTokens]).
    static func createInput(prompt: String, maxTokens: Int) throws -> [Tensor] {
        let utf8 = Array(prompt.utf8)
        var tokenIds = utf8.map { Int32(truncatingIfNeeded: $0) }
        if tokenIds.count > maxTokens {
            tokenIds = Array(tokenIds.prefix(maxTokens))
        } else {
            let padCount = maxTokens - tokenIds.count
            tokenIds.append(contentsOf: repeatElement(0, count: padCount))
        }
        let data = tokenIds.withUnsafeBufferPointer { buf in
            Data(buffer: buf)
        }
        let tensor = Tensor(data: data, dataType: BuiltinDataType.int32, shape: [1, maxTokens])
        return [tensor]
    }
}

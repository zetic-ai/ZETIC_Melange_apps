//
//  ModelInputSpec.swift
//  PromptGuard
//

import Foundation

/// Persisted spec for text-classification model input (editable in Diagnostics).
struct ModelInputSpec: Codable, Equatable {
    /// Maximum token count (sequence length).
    var maxTokens: Int
    /// Prompt template: "{user_input}" and optionally "{agent_output}" are replaced.
    var promptTemplate: String

    static let `default` = ModelInputSpec(
        maxTokens: 512,
        promptTemplate: "User: {user_input}\nAgent: {agent_output}"
    )

    func applied(userInput: String, agentOutput: String = "") -> String {
        promptTemplate
            .replacingOccurrences(of: "{user_input}", with: userInput)
            .replacingOccurrences(of: "{agent_output}", with: agentOutput)
    }
}

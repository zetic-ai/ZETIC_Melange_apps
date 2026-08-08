//
//  ModelInputSpecStore.swift
//  PromptGuard
//

import Foundation

/// Persists ModelInputSpec in UserDefaults.
final class ModelInputSpecStore {
    static let shared = ModelInputSpecStore()
    private let key = "promptguard_model_input_spec"
    private let defaults = UserDefaults.standard

    var spec: ModelInputSpec {
        get {
            guard let data = defaults.data(forKey: key),
                  let decoded = try? JSONDecoder().decode(ModelInputSpec.self, from: data) else {
                return .default
            }
            return decoded
        }
        set {
            if let data = try? JSONEncoder().encode(newValue) {
                defaults.set(data, forKey: key)
            }
        }
    }

    private init() {}
}

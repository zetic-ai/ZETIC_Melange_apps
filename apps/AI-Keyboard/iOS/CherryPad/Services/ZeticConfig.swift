import Foundation

/// Configuration for the ZETIC.ai Melange on-device models.
///
/// `personalKey` is the placeholder Melange Personal Access Token — run
/// `./adapt_mlange_key.sh` from the repo root to inject the real key (keeps it out
/// of git). Models resolve by `name`/`version` on the Melange dashboard.
enum ZeticConfig {
    static let personalKey = "dev_40e3948ba051485c9ccd827a2a17922f"
    static let modelVersion = 1

    /// The two on-device models the user can choose between. Both are lightweight.
    enum Quality: String, CaseIterable, Identifiable {
        case qwen   // default — Qwen3 0.6B (balanced, multilingual)
        case lfm    // LFM2.5 350M (lightest, fastest)

        var id: String { rawValue }

        /// Melange model name. Verified available for this key.
        var modelName: String {
            switch self {
            case .qwen: return "Qwen/Qwen3-0.6B"
            case .lfm:  return "Steve/LFM2.5_350M"
            }
        }

        var label: String {
            switch self {
            case .qwen: return "Qwen3 0.6B"
            case .lfm:  return "LFM2.5 350M"
            }
        }

        var detail: String {
            switch self {
            case .qwen: return "Balanced · multilingual · ~0.4 GB"
            case .lfm:  return "Lightest · fastest · ~0.3 GB"
            }
        }

        /// Qwen3 has a reasoning ("thinking") mode that must be suppressed with
        /// `/no_think`. LFM2.5 is not a reasoning model, so no directive is needed.
        var disablesThinking: Bool { self == .qwen }

        /// LFM2.5's dashboard recipe recommends RUN_ACCURACY; Qwen3-0.6B uses the
        /// SDK default (RUN_AUTO) — passing an explicit mode degraded its output.
        var usesAccuracyMode: Bool { self == .lfm }
    }

    private static let qualityKey = "cherrypad.quality"

    /// Persisted model choice; defaults to Qwen3-0.6B.
    static var quality: Quality {
        get { Quality(rawValue: UserDefaults.standard.string(forKey: qualityKey) ?? "") ?? .qwen }
        set { UserDefaults.standard.set(newValue.rawValue, forKey: qualityKey) }
    }
}

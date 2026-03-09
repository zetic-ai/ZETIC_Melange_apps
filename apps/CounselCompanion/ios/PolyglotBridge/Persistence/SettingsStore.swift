import Foundation

final class SettingsStore {
    private let defaults = UserDefaults.standard

    private enum Keys {
        static let theme = "pb_theme"
        static let defaultSource = "pb_source"
        static let defaultTarget = "pb_target"
        static let systemPrompt = "pb_system_prompt"
        static let modelReady = "pb_model_ready"
    }

    var modelReady: Bool {
        get { defaults.bool(forKey: Keys.modelReady) }
        set { defaults.set(newValue, forKey: Keys.modelReady) }
    }

    func load() -> SettingsData {
        let themeRaw = defaults.string(forKey: Keys.theme) ?? SettingsData.Theme.system.rawValue
        let theme = SettingsData.Theme(rawValue: themeRaw) ?? .system
        let source = defaults.string(forKey: Keys.defaultSource) ?? "en"
        let target = defaults.string(forKey: Keys.defaultTarget) ?? "es"
        let prompt = defaults.string(forKey: Keys.systemPrompt) ?? ""
        return SettingsData(theme: theme, defaultSource: source, defaultTarget: target, systemPrompt: prompt)
    }

    func save(_ settings: SettingsData) {
        defaults.set(settings.theme.rawValue, forKey: Keys.theme)
        defaults.set(settings.defaultSource, forKey: Keys.defaultSource)
        defaults.set(settings.defaultTarget, forKey: Keys.defaultTarget)
        defaults.set(settings.systemPrompt, forKey: Keys.systemPrompt)
    }
}

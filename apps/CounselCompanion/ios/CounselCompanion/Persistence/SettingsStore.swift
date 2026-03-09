import Foundation
import SwiftUI

final class SettingsStore: ObservableObject {
    @Published var theme: ColorScheme? = nil
    @Published var systemPrompt: String = defaultPrompt

    private enum Keys { static let theme = "theme_mode"; static let prompt = "system_prompt" }

    static let defaultPrompt = "You are a kind mental wellness companion. Listen actively, ask reflective questions, and avoid medical diagnosis."

    init() {
        let stored = UserDefaults.standard.string(forKey: Keys.theme)
        theme = switch stored { case "light": .light; case "dark": .dark; default: nil }
        systemPrompt = UserDefaults.standard.string(forKey: Keys.prompt) ?? Self.defaultPrompt
    }

    func setTheme(_ scheme: ColorScheme?) {
        theme = scheme
        let value: String? = switch scheme { case .some(.light): "light"; case .some(.dark): "dark"; default: nil }
        UserDefaults.standard.setValue(value, forKey: Keys.theme)
    }

    func setPrompt(_ prompt: String) {
        systemPrompt = prompt
        UserDefaults.standard.setValue(prompt, forKey: Keys.prompt)
    }
}

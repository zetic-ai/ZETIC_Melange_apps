//
//  AppTheme.swift
//  PromptGuard
//

import SwiftUI

enum AppTheme {
    static let accent = Color(red: 0.95, green: 0.65, blue: 0.2)
    static let danger = Color(red: 0.9, green: 0.25, blue: 0.2)
    static let safe = Color(red: 0.2, green: 0.75, blue: 0.5)
    static let cardBackground = Color(white: 0.12)
    static let cardBackgroundLight = Color(white: 0.95)
    static let textSecondary = Color(white: 0.55)
    static let textSecondaryLight = Color(white: 0.45)
}

struct ThemeEnvironmentKey: EnvironmentKey {
    static let defaultValue: ColorScheme = .dark
}

extension EnvironmentValues {
    var appColorScheme: ColorScheme {
        get { self[ThemeEnvironmentKey.self] }
        set { self[ThemeEnvironmentKey.self] = newValue }
    }
}

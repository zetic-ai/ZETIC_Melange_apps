//
//  AppTheme.swift
//  PromptGuard
//

import SwiftUI

enum AppTheme {
    static let zeticTeal = Color(red: 52/255, green: 169/255, blue: 163/255)
    static let brand = zeticTeal
    static let accent = zeticTeal
    static let danger = Color(red: 0.85, green: 0.25, blue: 0.25)
    static let safe = Color(red: 0.2, green: 0.7, blue: 0.5)

    static let background = Color(.systemGroupedBackground)
    static let cardBackground = Color(.secondarySystemGroupedBackground)
    static let textSecondary = Color(.secondaryLabel)

    /// Full-screen gradient (extends under status bar and home indicator).
    static var gradientBackground: LinearGradient {
        LinearGradient(
            colors: [
                Color(red: 0.07, green: 0.12, blue: 0.15), // deep teal base
                zeticTeal.opacity(0.9),                    // brand teal
                Color.blue.opacity(0.65)                   // soft blue blend
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
    static let tabBarMaterial: Material = .ultraThinMaterial
}

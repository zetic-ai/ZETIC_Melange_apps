import SwiftUI

/// Dark palette tuned to DeepL's iOS app (from the reference screenshots).
enum Theme {
    static let background = Color.black
    static let surface = Color(hex: 0x1B1B1D)        // the big translate card
    static let surfaceRaised = Color(hex: 0x2C2C2E)  // pills, language buttons
    static let surfaceRaisedHi = Color(hex: 0x3A3A3C)
    static let accent = Color(hex: 0x1A8FE3)         // bright blue: Paste, profile
    static let accentDeep = Color(hex: 0x0B63CE)     // selected segment
    static let textPrimary = Color.white
    static let textSecondary = Color(hex: 0x8E8E93)
    static let textTertiary = Color(hex: 0x636366)
    static let separator = Color(hex: 0x2E2E30)
    static let online = Color(hex: 0x32D74B)
}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: 1
        )
    }
}

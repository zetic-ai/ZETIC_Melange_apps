import SwiftUI

/// CherryPad's visual identity — a cherry-red take on MangoPad's bright, rounded,
/// chip-driven look. Light theme: warm off-white canvas, white cards, cherry-red
/// accents.
enum Theme {
    // MARK: Brand
    static let cherry = Color(hex: 0xD81E34)        // primary cherry red
    static let cherryDark = Color(hex: 0xA8132A)    // pressed / emphasis
    static let cherrySoft = Color(hex: 0xFBE3E6)    // tinted chip background

    // MARK: Surfaces
    static let background = Color(hex: 0xFFF7F4)     // warm off-white canvas
    static let surface = Color.white                 // cards
    static let surfaceMuted = Color(hex: 0xF1ECEA)   // input wells / inactive chips

    // MARK: Text
    static let textPrimary = Color(hex: 0x1C1A19)
    static let textSecondary = Color(hex: 0x837C78)
    static let onCherry = Color.white

    // MARK: Metrics
    static let cardRadius: CGFloat = 20
    static let chipRadius: CGFloat = 14
    static let cardShadow = Color.black.opacity(0.06)
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

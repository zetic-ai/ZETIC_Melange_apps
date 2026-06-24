import SwiftUI

/// Light medical-clean design tokens.
enum Theme {
    static let background = Color(red: 0.96, green: 0.97, blue: 0.99)   // soft off-white
    static let card = Color.white
    static let accent = Color(red: 0.18, green: 0.45, blue: 0.95)       // calm clinical blue
    static let accentSoft = Color(red: 0.18, green: 0.45, blue: 0.95).opacity(0.12)
    static let textPrimary = Color(red: 0.10, green: 0.13, blue: 0.20)
    static let textSecondary = Color(red: 0.42, green: 0.47, blue: 0.55)
    static let good = Color(red: 0.20, green: 0.72, blue: 0.48)
    static let fair = Color(red: 0.95, green: 0.62, blue: 0.18)
    static let poor = Color(red: 0.90, green: 0.30, blue: 0.30)

    static let cardRadius: CGFloat = 22
    static let cardShadow = Color.black.opacity(0.06)

    static func quality(_ q: Double) -> Color {
        if q >= 0.6 { return good }
        if q >= 0.3 { return fair }
        return poor
    }

    static func qualityLabel(_ q: Double) -> String {
        if q >= 0.6 { return "Good signal" }
        if q >= 0.3 { return "Fair signal" }
        return "Stabilizing…"
    }
}

/// Reusable soft card container.
struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .background(Theme.card)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
            .shadow(color: Theme.cardShadow, radius: 12, x: 0, y: 6)
    }
}

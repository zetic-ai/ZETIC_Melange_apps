import SwiftUI

/// Dark sports-analytics design tokens.
enum Theme {
    static let background = Color(red: 0.043, green: 0.055, blue: 0.075)
    static let card = Color(red: 0.086, green: 0.106, blue: 0.133)
    static let accent = Color(red: 0.275, green: 0.878, blue: 0.549)     // court green
    static let accentSoft = Color(red: 0.275, green: 0.878, blue: 0.549).opacity(0.14)
    static let ball = Color(red: 1.0, green: 0.62, blue: 0.15)           // ball orange
    static let leftSide = Color(red: 0.30, green: 0.80, blue: 1.0)       // left limbs
    static let rightSide = Color(red: 1.0, green: 0.45, blue: 0.55)      // right limbs
    static let torso = Color.white
    static let textPrimary = Color(red: 0.93, green: 0.95, blue: 0.97)
    static let textSecondary = Color(red: 0.55, green: 0.60, blue: 0.67)
    static let good = Color(red: 0.275, green: 0.878, blue: 0.549)
    static let poor = Color(red: 0.95, green: 0.35, blue: 0.35)

    static let cardRadius: CGFloat = 18
}

/// Reusable dark card container.
struct Card<Content: View>: View {
    @ViewBuilder var content: Content
    var body: some View {
        content
            .background(Theme.card)
            .clipShape(RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous))
    }
}

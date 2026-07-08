import SwiftUI

/// Futuristic / glassy visual language for the Skin Image Classification demo.
///
/// Dark, near-black base with a deep teal→indigo aurora, frosted glass cards
/// (`.ultraThinMaterial`), a cyan medical-tech accent, and thin hairline strokes.
/// Centralized so every surface reads as one polished, professional system.
enum Theme {

    // MARK: Palette
    static let accent  = Color(red: 0.36, green: 0.92, blue: 0.93)   // cyan glow
    static let accent2 = Color(red: 0.55, green: 0.62, blue: 1.00)   // periwinkle
    static let mint    = Color(red: 0.40, green: 0.92, blue: 0.70)
    static let amber   = Color(red: 1.00, green: 0.78, blue: 0.36)
    static let coral   = Color(red: 1.00, green: 0.45, blue: 0.48)

    static let ink     = Color.white
    static let inkSoft = Color.white.opacity(0.62)
    static let inkFaint = Color.white.opacity(0.40)

    // MARK: Backgrounds

    /// Full-screen aurora gradient backdrop. Place behind everything.
    static var background: some View {
        ZStack {
            Color(red: 0.04, green: 0.05, blue: 0.08).ignoresSafeArea()
            // Soft aurora blobs
            RadialGradient(
                colors: [accent.opacity(0.22), .clear],
                center: .topLeading, startRadius: 8, endRadius: 460
            ).ignoresSafeArea()
            RadialGradient(
                colors: [accent2.opacity(0.20), .clear],
                center: .bottomTrailing, startRadius: 8, endRadius: 520
            ).ignoresSafeArea()
        }
    }

    /// Hairline stroke used on glass edges for that crisp glassmorphism rim.
    static var hairline: LinearGradient {
        LinearGradient(
            colors: [.white.opacity(0.45), .white.opacity(0.06)],
            startPoint: .topLeading, endPoint: .bottomTrailing
        )
    }

    /// Brand gradient for primary text / accents.
    static var brandGradient: LinearGradient {
        LinearGradient(colors: [accent, accent2], startPoint: .leading, endPoint: .trailing)
    }

    // MARK: Type
    static func mono(_ size: CGFloat, _ weight: Font.Weight = .medium) -> Font {
        .system(size: size, weight: weight, design: .monospaced)
    }
}

// MARK: - Glass card modifier

extension View {
    /// Frosted glass panel with a hairline rim and soft depth shadow.
    func glassCard(cornerRadius: CGFloat = 24, padding: CGFloat = 18) -> some View {
        self
            .padding(padding)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(Theme.hairline, lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.35), radius: 18, x: 0, y: 10)
    }
}

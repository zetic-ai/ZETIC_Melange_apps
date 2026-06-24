import SwiftUI

/// Warm, editorial "private notes" palette shared with the VoiceVitals demo:
/// cream background, serif headlines, soft rounded cards, sage-green icon tiles,
/// deep-green accent. No brand colors — the design language matches `iOS/VoiceVitals`.
enum Theme {
    static let bg       = Color(red: 0.953, green: 0.945, blue: 0.918)  // warm cream
    static let card     = Color(red: 0.912, green: 0.903, blue: 0.872)  // soft warm-gray card
    static let cardAlt  = Color(red: 0.972, green: 0.966, blue: 0.945)  // lighter inner fill

    static let ink      = Color(red: 0.118, green: 0.118, blue: 0.106)  // near-black text
    static let inkSoft  = Color(red: 0.435, green: 0.427, blue: 0.400)  // muted secondary text

    static let tile     = Color(red: 0.792, green: 0.851, blue: 0.733)  // sage icon tile
    static let tileInk  = Color(red: 0.310, green: 0.447, blue: 0.282)  // deep green glyph
    static let dark     = Color(red: 0.168, green: 0.165, blue: 0.149)  // charcoal pill

    // Semantic aliases.
    static let accent   = tileInk        // primary action / selection
    static let positive = tileInk
    static let warn     = Color(red: 0.85, green: 0.62, blue: 0.26)
    static let danger   = Color(red: 0.83, green: 0.36, blue: 0.33)

    static let corner: CGFloat = 22
}

extension Font {
    /// Editorial serif display font.
    static func serif(_ size: CGFloat, _ weight: Font.Weight = .regular) -> Font {
        .system(size: size, weight: weight, design: .serif)
    }
}

extension View {
    /// Standard soft "card" container.
    func card(_ padding: CGFloat = 18) -> some View {
        self
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Theme.card)
            .clipShape(RoundedRectangle(cornerRadius: Theme.corner, style: .continuous))
    }
}

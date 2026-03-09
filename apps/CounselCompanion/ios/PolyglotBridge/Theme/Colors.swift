import SwiftUI

enum AppColors {
    static let accent = Color(red: 0.18, green: 0.61, blue: 0.78)
    static let accentGradient = LinearGradient(colors: [Color(red: 0.18, green: 0.61, blue: 0.78), Color(red: 0.32, green: 0.36, blue: 0.80)], startPoint: .leading, endPoint: .trailing)
    static let background = Color("Background", bundle: .main)
    static let card = Color("Card", bundle: .main)
}

struct GlassBackground: View {
    var body: some View {
        RoundedRectangle(cornerRadius: 20, style: .continuous)
            .fill(.ultraThinMaterial)
            .overlay(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .strokeBorder(Color.white.opacity(0.08), lineWidth: 1)
            )
    }
}

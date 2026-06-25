import SwiftUI

/// Compact pill conveying the current guidance / signal quality.
struct SignalQualityBadge: View {
    let quality: Double
    let faceFound: Bool
    let lowLight: Bool

    private var label: String {
        if !faceFound { return "Center your face" }
        if lowLight { return "More light" }
        return Theme.qualityLabel(quality)
    }

    private var color: Color {
        if !faceFound || lowLight { return Theme.poor }
        return Theme.quality(quality)
    }

    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 9, height: 9)
            Text(label)
                .font(.system(size: 14, weight: .semibold, design: .rounded))
                .foregroundStyle(Theme.textPrimary)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

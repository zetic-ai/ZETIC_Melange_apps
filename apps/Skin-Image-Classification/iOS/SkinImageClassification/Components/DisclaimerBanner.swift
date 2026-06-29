import SwiftUI

/// Persistent, non-dismissible safety notice. Rendered unconditionally on the
/// results screen so the user always sees it even if the model omits its own line.
struct DisclaimerBanner: View {
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: "info.circle.fill")
                .font(.system(size: 14))
                .foregroundStyle(Theme.amber)
            Text("Demo only — not a medical device. This is not a diagnosis. Consult a healthcare professional.")
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Theme.inkSoft)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 14)
        .padding(.vertical, 11)
        .background(Color.white.opacity(0.05), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(Theme.amber.opacity(0.25), lineWidth: 1)
        )
    }
}

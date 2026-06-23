import SwiftUI

/// Live network status pill — reflects the *real* connection. Online is incidental;
/// the demo's point is that translation still works when this reads "Offline".
struct LiveStatusBadge: View {
    let isOnline: Bool

    private var tint: Color { isOnline ? Theme.online : Theme.textSecondary }

    var body: some View {
        HStack(spacing: 5) {
            Circle()
                .fill(tint)
                .frame(width: 7, height: 7)
            Text(isOnline ? "Online" : "Offline")
                .font(.system(size: 11, weight: .semibold))
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(tint.opacity(0.14), in: Capsule())
    }
}

/// Tasteful "Powered by Zetic" footnote.
struct PoweredByZetic: View {
    var body: some View {
        HStack(spacing: 4) {
            Text("On-device translation · powered by")
                .foregroundStyle(Theme.textTertiary)
            Text("Zetic")
                .foregroundStyle(Theme.textSecondary)
                .fontWeight(.semibold)
        }
        .font(.system(size: 10))
    }
}

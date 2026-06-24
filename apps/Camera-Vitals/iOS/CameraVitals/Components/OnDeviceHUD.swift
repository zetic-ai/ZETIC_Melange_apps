import SwiftUI

/// Reinforces the pitch: everything runs on-device on the Neural Engine, with live latency.
struct OnDeviceHUD: View {
    let latencyMs: Double

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "cpu")
                .font(.system(size: 12, weight: .bold))
            Text("On-device · NPU")
                .font(.system(size: 12, weight: .bold, design: .rounded))
            if latencyMs > 0 {
                Text("· \(Int(latencyMs.rounded())) ms")
                    .font(.system(size: 12, weight: .medium, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(Theme.textSecondary)
            }
        }
        .foregroundStyle(Theme.accent)
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(.ultraThinMaterial, in: Capsule())
    }
}

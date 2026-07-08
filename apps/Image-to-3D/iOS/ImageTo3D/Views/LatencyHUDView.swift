import SwiftUI

/// Compact on-device stats overlay: per-stage latency.
struct LatencyHUDView: View {
    let latency: ImageTo3DViewModel.Latency

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            row("load", latency.modelLoadMs)
            row("depth", latency.depthMs)
            row("mesh", latency.meshMs)
        }
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 8))
    }

    private func row(_ label: String, _ ms: Double) -> some View {
        HStack(spacing: 4) {
            Text(label).foregroundColor(.secondary)
            Spacer(minLength: 8)
            Text(ms >= 1000 ? String(format: "%.1f s", ms / 1000)
                            : String(format: "%.0f ms", ms))
        }
    }
}

import SwiftUI

/// The demo's selling point: live per-model latency, sustained FPS, and peak memory.
struct BenchmarkHUD: View {
    let stats: BenchmarkSnapshot
    let liftAvailable: Bool
    let mode: ClipFrameSource.Mode

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "cpu")
                    .font(.system(size: 11, weight: .bold))
                Text("On-device · Melange")
                    .font(.system(size: 11, weight: .bold, design: .rounded))
                Spacer()
                Text(mode == .benchmark ? "sustained" : "realtime")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(Theme.textSecondary)
            }
            .foregroundStyle(Theme.accent)

            VStack(spacing: 4) {
                HStack {
                    Text("memory")
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .foregroundStyle(Theme.textSecondary)
                    Spacer()
                    Text(String(format: "%.0f MB · peak %.0f", stats.memoryMB, stats.peakMemoryMB))
                        .font(.system(size: 11, weight: .medium, design: .rounded))
                        .monospacedDigit()
                        .foregroundStyle(Theme.textPrimary)
                }
                latencyRow("YOLO26n", stats.detectMs)
                latencyRow("RTMPose-s", stats.poseMs)
                if liftAvailable {
                    latencyRow("3D lift", stats.liftMs)
                }
                latencyRow("pipeline", stats.totalMs, emphasized: true)
            }
        }
        .padding(12)
        .frame(width: 220)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14, style: .continuous))
    }

    private func latencyRow(_ label: String, _ ms: Double, emphasized: Bool = false) -> some View {
        HStack {
            Text(label)
                .font(.system(size: 11, weight: emphasized ? .bold : .medium, design: .rounded))
                .foregroundStyle(emphasized ? Theme.textPrimary : Theme.textSecondary)
            Spacer()
            Text(ms > 0 ? String(format: "%.1f ms", ms) : "—")
                .font(.system(size: 11, weight: emphasized ? .bold : .medium, design: .rounded))
                .monospacedDigit()
                .foregroundStyle(emphasized ? Theme.accent : Theme.textPrimary)
        }
    }
}

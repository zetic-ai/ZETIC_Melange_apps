import SwiftUI

/// Shown while the Melange model downloads & compiles for the Neural Engine.
struct DownloadView: View {
    let progress: Float

    var body: some View {
        VStack(spacing: 26) {
            ZStack {
                Circle()
                    .stroke(Theme.accentSoft, lineWidth: 10)
                    .frame(width: 120, height: 120)
                Circle()
                    .trim(from: 0, to: CGFloat(max(0.02, progress)))
                    .stroke(Theme.accent, style: StrokeStyle(lineWidth: 10, lineCap: .round))
                    .frame(width: 120, height: 120)
                    .rotationEffect(.degrees(-90))
                    .animation(.easeOut(duration: 0.3), value: progress)
                Image(systemName: "cpu")
                    .font(.system(size: 34, weight: .semibold))
                    .foregroundStyle(Theme.accent)
            }
            VStack(spacing: 6) {
                Text("Preparing on-device model")
                    .font(.system(size: 20, weight: .bold, design: .rounded))
                    .foregroundStyle(Theme.textPrimary)
                Text(progress > 0 ? "\(Int(progress * 100))%" : "Optimizing for the Neural Engine…")
                    .font(.system(size: 15))
                    .monospacedDigit()
                    .foregroundStyle(Theme.textSecondary)
            }
        }
    }
}

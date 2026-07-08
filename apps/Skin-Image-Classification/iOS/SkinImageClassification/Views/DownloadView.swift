import SwiftUI

/// First-run gate that loads the two on-device models. Shows progress for each;
/// the classifier gates entry (RootView advances automatically once it's ready),
/// while MedGemma keeps loading in the background.
struct DownloadView: View {
    @EnvironmentObject private var vm: DiagnosisViewModel

    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            BrandMark()

            VStack(spacing: 6) {
                Text("Preparing on-device AI")
                    .font(.system(size: 21, weight: .semibold))
                    .foregroundStyle(Theme.ink)
                Text("The skin vision model is loading directly onto this device. Nothing is sent to the cloud.")
                    .font(.system(size: 13.5))
                    .foregroundStyle(Theme.inkSoft)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }

            VStack(spacing: 14) {
                ModelLoadRow(
                    title: "Skin Vision Model",
                    subtitle: "ViT classifier · 7 lesion types",
                    icon: "viewfinder",
                    phase: vm.classifierPhase
                )
            }
            .glassCard(cornerRadius: 26, padding: 18)
            .padding(.horizontal, 22)

            if let err = vm.classifierPhase.errorMessage {
                VStack(spacing: 10) {
                    Text(err)
                        .font(.system(size: 12.5))
                        .foregroundStyle(Theme.coral)
                        .multilineTextAlignment(.center)
                    Button("Try again") { vm.retryLoad() }
                        .font(.system(size: 14, weight: .semibold))
                        .tint(Theme.accent)
                }
                .padding(.horizontal, 28)
            }

            Spacer()

            Text("Powered by ZETIC Melange")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Theme.inkFaint)
                .tracking(1.0)
                .padding(.bottom, 18)
        }
    }
}

/// One row in the download card with a status-driven trailing indicator.
private struct ModelLoadRow: View {
    let title: String
    let subtitle: String
    let icon: String
    let phase: LoadPhase

    var body: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle().fill(Color.white.opacity(0.06)).frame(width: 42, height: 42)
                Image(systemName: icon)
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(Theme.brandGradient)
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.system(size: 15, weight: .semibold)).foregroundStyle(Theme.ink)
                Text(statusText).font(.system(size: 12)).foregroundStyle(Theme.inkSoft)
            }
            Spacer()
            indicator
        }
    }

    private var statusText: String {
        switch phase {
        case .idle, .preparing: return "Preparing…"
        case .downloading(let p): return "Downloading \(Int(p * 100))%"
        case .ready: return "Ready"
        case .failed: return "Failed"
        }
    }

    @ViewBuilder private var indicator: some View {
        switch phase {
        case .ready:
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 20)).foregroundStyle(Theme.mint)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 18)).foregroundStyle(Theme.coral)
        case .downloading(let p):
            ZStack {
                Circle().stroke(Color.white.opacity(0.1), lineWidth: 3).frame(width: 24, height: 24)
                Circle().trim(from: 0, to: CGFloat(p))
                    .stroke(Theme.accent, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(-90)).frame(width: 24, height: 24)
            }
        case .idle, .preparing:
            ProgressView().tint(Theme.accent).scaleEffect(0.85)
        }
    }
}

/// Animated brand glyph for the download / capture headers.
struct BrandMark: View {
    @State private var pulse = false
    var body: some View {
        ZStack {
            Circle()
                .fill(Theme.brandGradient)
                .frame(width: 84, height: 84)
                .blur(radius: pulse ? 14 : 8)
                .opacity(0.5)
            Image(systemName: "waveform.path.ecg.rectangle")
                .font(.system(size: 36, weight: .light))
                .foregroundStyle(Theme.ink)
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 1.8).repeatForever(autoreverses: true)) { pulse.toggle() }
        }
    }
}

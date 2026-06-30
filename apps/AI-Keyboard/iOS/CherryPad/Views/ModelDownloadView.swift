import SwiftUI

/// Compact model-status banner shown at the top of the compose screen while the
/// on-device model downloads/prepares. Hidden once the model is ready.
struct ModelStatusBanner: View {
    @ObservedObject var llm: LLMService

    var body: some View {
        switch llm.phase {
        case .ready:
            EmptyView()
        case .idle, .preparing:
            banner(icon: "arrow.down.circle", text: "Preparing on-device model…", showSpinner: true)
        case .downloading(let p):
            banner(icon: "arrow.down.circle", text: "Downloading model… \(Int(p * 100))%", showSpinner: true)
        case .failed(let message):
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(Theme.cherryDark)
                Text(message).font(.system(size: 13)).foregroundStyle(Theme.textPrimary)
                Spacer()
                Button("Retry") { Task { await llm.ensureLoaded() } }
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(Theme.cherry)
            }
            .padding(12)
            .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.cherrySoft))
        }
    }

    private func banner(icon: String, text: String, showSpinner: Bool) -> some View {
        HStack(spacing: 10) {
            if showSpinner {
                ProgressView().controlSize(.small).tint(Theme.cherry)
            } else {
                Image(systemName: icon).foregroundStyle(Theme.cherry)
            }
            Text(text).font(.system(size: 13, weight: .medium)).foregroundStyle(Theme.textSecondary)
            Spacer()
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 14, style: .continuous).fill(Theme.surfaceMuted))
    }
}

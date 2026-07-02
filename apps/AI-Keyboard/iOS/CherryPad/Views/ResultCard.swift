import SwiftUI

/// The streamed result card with Retake / Apply, echoing MangoPad's result panel.
struct ResultCard: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 6) {
                Image(systemName: model.task.symbol)
                    .font(.system(size: 12, weight: .bold))
                Text("\(model.task.title) result")
                    .font(.system(size: 13, weight: .bold))
                Spacer()
                if model.isGenerating {
                    ProgressView().controlSize(.small).tint(Theme.cherry)
                }
            }
            .foregroundStyle(Theme.textSecondary)

            Group {
                if let error = model.errorMessage {
                    Text(error)
                        .font(.system(size: 15))
                        .foregroundStyle(Theme.cherryDark)
                } else if model.resultText.isEmpty && model.isGenerating {
                    Text("Thinking…")
                        .font(.system(size: 15))
                        .foregroundStyle(Theme.textSecondary)
                        .shimmer()
                } else {
                    Text(model.resultText)
                        .font(.system(size: 16))
                        .foregroundStyle(Theme.textPrimary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }

            if !model.isGenerating && model.errorMessage == nil && !model.resultText.isEmpty {
                HStack(spacing: 10) {
                    Button(action: model.retake) {
                        Label("Retake", systemImage: "arrow.counterclockwise")
                            .font(.system(size: 14, weight: .semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 11)
                            .foregroundStyle(Theme.textPrimary)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.chipRadius, style: .continuous)
                                    .fill(Theme.surfaceMuted)
                            )
                    }
                    .buttonStyle(.plain)

                    Button(action: model.apply) {
                        Label(model.didApply ? "Copied" : "Apply",
                              systemImage: model.didApply ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 14, weight: .semibold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 11)
                            .foregroundStyle(Theme.onCherry)
                            .background(
                                RoundedRectangle(cornerRadius: Theme.chipRadius, style: .continuous)
                                    .fill(Theme.cherry)
                            )
                    }
                    .buttonStyle(.plain)
                }

                if model.fromKeyboard || model.didApply {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                        Text("Copied! Switch back to your app and **paste** — or reopen the CherryPad keyboard and tap **Insert result**.")
                            .font(.system(size: 13))
                    }
                    .foregroundStyle(Theme.cherryDark)
                    .padding(10)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 12, style: .continuous).fill(Theme.cherrySoft))
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: Theme.cardRadius, style: .continuous)
                .fill(Theme.surface)
                .shadow(color: Theme.cardShadow, radius: 12, y: 4)
        )
    }
}

/// Subtle shimmer for the "Thinking…" placeholder.
private struct Shimmer: ViewModifier {
    @State private var phase: CGFloat = -1
    func body(content: Content) -> some View {
        content
            .opacity(0.5)
            .overlay(
                LinearGradient(
                    colors: [.clear, Theme.cherry.opacity(0.25), .clear],
                    startPoint: .leading, endPoint: .trailing
                )
                .offset(x: phase * 120)
                .mask(content)
            )
            .onAppear {
                withAnimation(.linear(duration: 1.1).repeatForever(autoreverses: false)) {
                    phase = 1.5
                }
            }
    }
}

extension View {
    func shimmer() -> some View { modifier(Shimmer()) }
}

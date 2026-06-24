import SwiftUI

/// Result state (reference screenshot #3): source on top, streamed translation on the
/// bottom, with an action footer. Tapping the source returns to editing.
struct ResultView: View {
    @ObservedObject var vm: TranslationViewModel
    let onEditSource: () -> Void

    var body: some View {
        VStack(spacing: 0) {
            // Source (top half)
            VStack(alignment: .leading, spacing: 12) {
                Button(action: onEditSource) {
                    ScrollView {
                        Text(vm.sourceText)
                            .font(.system(size: 20, weight: .regular))
                            .foregroundStyle(Theme.textPrimary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)

                iconButton("speaker.wave.2") { vm.speak(vm.sourceText, language: vm.source) }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(.bottom, 14)

            Rectangle()
                .fill(Theme.separator)
                .frame(height: 1)

            // Translation (bottom half)
            VStack(alignment: .leading, spacing: 12) {
                ScrollView {
                    translationText
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)

                footer
            }
            .padding(.top, 16)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
    }

    @ViewBuilder
    private var translationText: some View {
        if vm.translatedText.isEmpty && vm.isTranslating {
            HStack(spacing: 6) {
                ProgressView().tint(Theme.textSecondary)
                Text("Translating on device…")
                    .font(.system(size: 15))
                    .foregroundStyle(Theme.textSecondary)
            }
        } else {
            (Text(vm.translatedText)
                + Text(vm.isTranslating ? " ▍" : "").foregroundColor(Theme.accent))
                .font(.system(size: 20, weight: .regular))
                .foregroundStyle(Theme.textPrimary)
        }
    }

    private var footer: some View {
        HStack(spacing: 20) {
            iconButton("speaker.wave.2") { vm.speak(vm.translatedText, language: vm.target) }

            Text("Alternatives")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Theme.textSecondary)
                .padding(.horizontal, 13)
                .padding(.vertical, 7)
                .background(Theme.surfaceRaised, in: Capsule())

            Spacer()

            iconButton("bookmark") {}                       // placeholder
            ShareLink(item: vm.translatedText) {
                Image(systemName: "square.and.arrow.up")
                    .font(.system(size: 18, weight: .regular))
                    .foregroundStyle(Theme.textPrimary)
            }
            .disabled(vm.translatedText.isEmpty)
            iconButton("doc.on.doc") { vm.copyTranslation() }
        }
    }

    private func iconButton(_ name: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: name)
                .font(.system(size: 18, weight: .regular))
                .foregroundStyle(Theme.textPrimary)
        }
        .buttonStyle(.plain)
    }
}

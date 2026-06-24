import SwiftUI

/// Landing state (reference screenshot #4): prompt text, Paste, and the voice/image source
/// buttons. Tapping the body starts editing.
struct IdleView: View {
    @ObservedObject var vm: TranslationViewModel
    let onActivate: () -> Void
    let onVoice: () -> Void
    let onImage: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button(action: onActivate) {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Type, paste, talk, or snap a photo to translate")
                        .font(.system(size: 24, weight: .regular))
                        .foregroundStyle(Theme.textSecondary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)

                    PasteButton(enabled: vm.modelState == .ready) {
                        vm.paste()
                        onActivate()
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Spacer(minLength: 22)

            // Offline input sources: voice (speech-to-text) and image (OCR).
            HStack(spacing: 16) {
                sourceButton("mic.fill", action: onVoice)
                sourceButton("camera.fill", action: onImage)
                Spacer()
            }
        }
    }

    private func sourceButton(_ icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(Theme.accent)
                .frame(width: 53, height: 53)
                .background(Theme.surfaceRaised, in: Circle())
        }
        .buttonStyle(.plain)
    }
}

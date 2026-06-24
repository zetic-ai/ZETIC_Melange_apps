import SwiftUI

/// The persistent bottom language selector: source pill · swap · target pill.
struct LanguageBar: View {
    let source: Language
    let target: Language
    let onTapSource: () -> Void
    let onTapTarget: () -> Void
    let onSwap: () -> Void

    var body: some View {
        HStack(spacing: 10) {
            pill(source.englishName, action: onTapSource)

            Button(action: onSwap) {
                Image(systemName: source.isDetect ? "arrow.right" : "arrow.left.arrow.right")
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(Theme.textPrimary)
                    .frame(width: 32, height: 32)
            }

            pill(target.englishName, action: onTapTarget)
        }
    }

    private func pill(_ title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(Theme.surfaceRaised, in: RoundedRectangle(cornerRadius: 13, style: .continuous))
        }
    }
}

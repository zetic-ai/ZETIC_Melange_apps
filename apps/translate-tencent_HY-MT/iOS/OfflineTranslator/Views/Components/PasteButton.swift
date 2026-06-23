import SwiftUI

/// The bright-blue "Paste" button from DeepL's input screen.
struct PasteButton: View {
    var enabled: Bool = true
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 7) {
                Image(systemName: "doc.on.clipboard")
                    .font(.system(size: 15, weight: .semibold))
                Text("Paste")
                    .font(.system(size: 16, weight: .semibold))
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 16)
            .padding(.vertical, 11)
            .background(
                (enabled ? Theme.accent : Theme.accent.opacity(0.4)),
                in: RoundedRectangle(cornerRadius: 11, style: .continuous)
            )
        }
        .disabled(!enabled)
    }
}

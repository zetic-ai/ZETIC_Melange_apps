import SwiftUI

/// Text-entry state (reference screenshot #2): an editable field with placeholder
/// and a Paste button while empty.
struct EditingView: View {
    @ObservedObject var vm: TranslationViewModel
    @FocusState.Binding var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            ZStack(alignment: .topLeading) {
                if vm.sourceText.isEmpty {
                    Text("Type or paste here to translate")
                        .font(.system(size: 24, weight: .regular))
                        .foregroundStyle(Theme.textSecondary)
                        .padding(.top, 8)
                        .allowsHitTesting(false)
                }
                TextEditor(text: $vm.sourceText)
                    .font(.system(size: 24, weight: .regular))
                    .foregroundStyle(Theme.textPrimary)
                    .tint(Theme.accent)
                    .scrollContentBackground(.hidden)
                    .background(.clear)
                    .focused($focused)
                    .frame(minHeight: 130, maxHeight: .infinity, alignment: .topLeading)
            }

            if vm.sourceText.isEmpty {
                PasteButton(enabled: vm.modelState == .ready) { vm.paste() }
            }

            Spacer(minLength: 0)
        }
    }
}

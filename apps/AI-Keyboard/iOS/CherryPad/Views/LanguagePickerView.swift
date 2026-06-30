import SwiftUI

/// Searchable target-language picker sheet.
struct LanguagePickerView: View {
    let selected: Language
    let onSelect: (Language) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    private var filtered: [Language] {
        guard !query.isEmpty else { return Language.all }
        return Language.all.filter {
            $0.englishName.localizedCaseInsensitiveContains(query)
                || $0.nativeName.localizedCaseInsensitiveContains(query)
        }
    }

    var body: some View {
        NavigationStack {
            List(filtered) { language in
                Button {
                    onSelect(language)
                    dismiss()
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(language.englishName)
                                .font(.system(size: 16, weight: .medium))
                                .foregroundStyle(Theme.textPrimary)
                            if language.nativeName != language.englishName {
                                Text(language.nativeName)
                                    .font(.system(size: 12))
                                    .foregroundStyle(Theme.textSecondary)
                            }
                        }
                        Spacer()
                        if language.id == selected.id {
                            Image(systemName: "checkmark")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundStyle(Theme.cherry)
                        }
                    }
                }
                .listRowBackground(Theme.surface)
            }
            .listStyle(.plain)
            .navigationTitle("Translate to")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.foregroundStyle(Theme.cherry)
                }
            }
            .searchable(text: $query, prompt: "Search languages")
        }
    }
}

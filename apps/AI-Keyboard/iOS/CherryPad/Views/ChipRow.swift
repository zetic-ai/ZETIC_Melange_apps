import SwiftUI

/// A single selectable chip (tone / stance / language trigger).
struct Chip: View {
    let title: String
    let isSelected: Bool
    var icon: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if let icon {
                    Image(systemName: icon).font(.system(size: 11, weight: .semibold))
                }
                Text(title).font(.system(size: 13, weight: .semibold))
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .foregroundStyle(isSelected ? Theme.cherryDark : Theme.textSecondary)
            .background(
                Capsule().fill(isSelected ? Theme.cherrySoft : Theme.surfaceMuted)
            )
            .overlay(
                Capsule().stroke(isSelected ? Theme.cherry.opacity(0.5) : .clear, lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

/// Context-sensitive option row beneath the action bar: tones for Rewrite, stance
/// for Reply, a language trigger for Translate, nothing for Grammar.
struct ChipRow: View {
    let task: KeyboardTask
    @Binding var tone: Tone
    @Binding var stance: Stance
    @Binding var targetLanguage: Language
    var onPickLanguage: () -> Void

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                switch task {
                case .rewrite:
                    ForEach(Tone.allCases) { t in
                        Chip(title: t.label, isSelected: t == tone) { tone = t }
                    }
                case .reply:
                    ForEach(Stance.allCases) { s in
                        Chip(title: s.label, isSelected: s == stance) { stance = s }
                    }
                case .translate:
                    Chip(title: targetLanguage.englishName, isSelected: true, icon: "globe", action: onPickLanguage)
                case .grammar:
                    EmptyView()
                }
            }
            .padding(.horizontal, 2)
        }
    }
}

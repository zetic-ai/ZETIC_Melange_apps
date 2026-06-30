import SwiftUI

/// The four AI actions, rendered as a row of pill buttons (MangoPad's Rewrite /
/// Reply / Translate / Grammar bar). The selected action is filled cherry-red.
struct ActionBar: View {
    @Binding var selection: KeyboardTask

    var body: some View {
        HStack(spacing: 8) {
            ForEach(KeyboardTask.allCases) { task in
                let isSelected = task == selection
                Button {
                    withAnimation(.snappy(duration: 0.18)) { selection = task }
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: task.symbol)
                            .font(.system(size: 15, weight: .semibold))
                        Text(task.title)
                            .font(.system(size: 12, weight: .semibold))
                            .lineLimit(1)
                            .minimumScaleFactor(0.8)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .foregroundStyle(isSelected ? Theme.onCherry : Theme.textSecondary)
                    .background(
                        RoundedRectangle(cornerRadius: Theme.chipRadius, style: .continuous)
                            .fill(isSelected ? Theme.cherry : Theme.surfaceMuted)
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }
}

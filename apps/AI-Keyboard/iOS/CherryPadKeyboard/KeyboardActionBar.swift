import SwiftUI

/// Palette local to the keyboard extension (it can't see the app's `Theme`).
enum KB {
    static let cherry = Color(red: 0.847, green: 0.118, blue: 0.204)
    static let cherryDark = Color(red: 0.659, green: 0.075, blue: 0.165)
    static let cherrySoft = Color(red: 0.984, green: 0.890, blue: 0.902)
    static let background = Color(.systemGray5)
    static let keyFill = Color(.systemBackground)
    static let specialFill = Color(.systemGray3)
    static let textPrimary = Color(.label)
    static let textSecondary = Color(.secondaryLabel)
}

/// The AI bar above the keys. Three states: idle (4 action buttons), processing
/// ("Thinking…"), and result-ready (preview + "Insert result"). Everything runs in
/// the keyboard — no app round-trip.
struct KeyboardActionBar: View {
    @ObservedObject var state: KeyboardState

    var body: some View {
        VStack(spacing: 6) {
            if let banner = state.banner {
                Text(banner)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(KB.cherry)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
            }

            if state.processing {
                processingRow
            } else if let result = state.resultText {
                resultRow(result)
            } else {
                actionsRow
            }
        }
    }

    private var actionsRow: some View {
        HStack(spacing: 6) {
            ForEach(KeyboardTask.allCases) { task in
                Button {
                    state.banner = nil
                    state.controller?.runAction(task)
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: task.symbol).font(.system(size: 16, weight: .semibold))
                        Text(task.title).font(.system(size: 11, weight: .semibold))
                            .lineLimit(1).minimumScaleFactor(0.8)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 11)
                    .foregroundStyle(KB.cherry)
                    .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(KB.keyFill))
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var processingRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small).tint(KB.cherry)
            Text(state.statusText ?? "Thinking…")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(KB.textSecondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 8)
        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(KB.keyFill))
    }

    private func resultRow(_ result: String) -> some View {
        VStack(spacing: 8) {
            ScrollView {
                Text(result)
                    .font(.system(size: 15))
                    .foregroundStyle(KB.textPrimary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 92)
            .padding(.horizontal, 12).padding(.vertical, 8)
            .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(KB.keyFill))

            HStack(spacing: 8) {
                Button { state.controller?.dismissResult() } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundStyle(KB.textSecondary)
                        .frame(width: 44)
                        .padding(.vertical, 9)
                        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(KB.specialFill))
                }.buttonStyle(.plain)

                if let task = state.activeTask {
                    Button { state.controller?.runAction(task) } label: {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 13, weight: .bold))
                            .foregroundStyle(KB.textPrimary)
                            .frame(width: 44)
                            .padding(.vertical, 9)
                            .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(KB.specialFill))
                    }.buttonStyle(.plain)
                }

                Button { state.controller?.insertResult() } label: {
                    Label("Insert result", systemImage: "arrow.down.doc.fill")
                        .font(.system(size: 14, weight: .bold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 9)
                        .foregroundStyle(.white)
                        .background(RoundedRectangle(cornerRadius: 10, style: .continuous).fill(KB.cherry))
                }.buttonStyle(.plain)
            }
        }
    }
}

import SwiftUI

/// Palette local to the keyboard extension (it can't see the app's `Theme`).
enum KB {
    static let cherry = Color(red: 0.847, green: 0.118, blue: 0.204)
    static let background = Color(.systemGray5)
    static let keyFill = Color(.systemBackground)
    static let specialFill = Color(.systemGray3)
    static let textPrimary = Color(.label)
}

/// The AI action bar above the keys: the four CherryPad actions, plus an
/// "Insert result" pill when the app has published a result.
struct KeyboardActionBar: View {
    @ObservedObject var state: KeyboardState

    var body: some View {
        VStack(spacing: 6) {
            if let banner = state.banner {
                Text(banner)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(KB.cherry)
                    .frame(maxWidth: .infinity)
            }

            HStack(spacing: 6) {
                if state.resultAvailable {
                    Button {
                        state.controller?.insertResult()
                    } label: {
                        Label("Insert result", systemImage: "arrow.down.doc.fill")
                            .font(.system(size: 13, weight: .bold))
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 9)
                            .foregroundStyle(.white)
                            .background(Capsule().fill(KB.cherry))
                    }
                    .buttonStyle(.plain)
                } else {
                    ForEach(KeyboardTask.allCases) { task in
                        Button {
                            state.banner = nil
                            state.controller?.runAction(task)
                        } label: {
                            VStack(spacing: 2) {
                                Image(systemName: task.symbol).font(.system(size: 13, weight: .semibold))
                                Text(task.title).font(.system(size: 10, weight: .semibold))
                                    .lineLimit(1).minimumScaleFactor(0.8)
                            }
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 6)
                            .foregroundStyle(KB.cherry)
                            .background(
                                RoundedRectangle(cornerRadius: 10, style: .continuous)
                                    .fill(KB.keyFill)
                            )
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

import SwiftUI

/// Design tokens + palette local to the keyboard extension (it can't see the app's
/// `Theme`). One radius scale + spacing/height rhythm keeps every state consistent.
enum KB {
    // Colors (semantic → dark-mode safe; cherry fixed)
    static let cherry = Color(red: 0.847, green: 0.118, blue: 0.204)
    static let cherryDark = Color(red: 0.659, green: 0.075, blue: 0.165)
    static let cherrySoft = Color(red: 0.984, green: 0.890, blue: 0.902)
    static let background = Color(.systemGray5)
    static let keyFill = Color(.systemBackground)
    static let specialFill = Color(.systemGray3)
    static let textPrimary = Color(.label)
    static let textSecondary = Color(.secondaryLabel)

    // Radii
    static let radiusKey: CGFloat = 5       // QWERTY keycaps
    static let radiusControl: CGFloat = 12  // every AI card / pill / button

    // Spacing
    static let sideMargin: CGFloat = 4
    static let sectionGap: CGFloat = 8
    static let barGap: CGFloat = 6
    static let keyGapH: CGFloat = 5
    static let keyGapV: CGFloat = 9

    // Heights
    static let keyHeight: CGFloat = 42
    static let actionButtonH: CGFloat = 52   // idle chips + processing pill
    static let controlHeight: CGFloat = 48   // result buttons (all equal)
    static let secondaryWidth: CGFloat = 52
    static let resultMinHeight: CGFloat = 38
    static let resultMaxHeight: CGFloat = 168

    // Soft system-style key shadow
    static let keyShadow = Color.black.opacity(0.16)

    /// The keyboard's FIXED height = idle content (action bar + QWERTY + paddings).
    /// Every state fits inside this; result/processing take over the area (keys hide).
    static var keyboardHeight: CGFloat {
        let keys = 4 * keyHeight + 3 * keyGapV          // 3 letter rows + bottom row, 3 gaps
        let content = actionButtonH + sectionGap + keys // action bar + gap + keys
        return sectionGap + content + 4                 // top pad + content + bottom pad
    }
}

/// The AI bar above the keys. Three states — idle (4 chips), processing (spinner
/// pill), result (content-hugging preview + one aligned ✕/↺/Insert group).
struct KeyboardActionBar: View {
    @ObservedObject var state: KeyboardState

    /// Result & processing take over the whole keyboard area; idle is a compact bar.
    private var isPanel: Bool { state.processing || state.resultText != nil }

    var body: some View {
        VStack(spacing: KB.barGap) {
            if let banner = state.banner {
                Text(banner)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(KB.cherryDark)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
                    .background(Capsule().fill(KB.cherrySoft))
            }

            if state.processing {
                processingRow
            } else if let result = state.resultText {
                resultRow(result)
            } else {
                actionsRow
            }
        }
        .frame(maxHeight: isPanel ? .infinity : nil)
    }

    // MARK: Idle

    private var actionsRow: some View {
        HStack(spacing: KB.barGap) {
            ForEach(KeyboardTask.allCases) { task in
                Button {
                    state.banner = nil
                    state.controller?.runAction(task)
                } label: {
                    VStack(spacing: 3) {
                        Image(systemName: task.symbol).font(.system(size: 17, weight: .semibold))
                        Text(task.title).font(.system(size: 11, weight: .semibold))
                            .lineLimit(1).minimumScaleFactor(0.75)
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: KB.actionButtonH)
                    .foregroundStyle(KB.cherry)
                    .background(card(KB.keyFill))
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: Processing

    private var processingRow: some View {
        HStack(spacing: 8) {
            ProgressView().controlSize(.small).tint(KB.cherry)
            Text(state.statusText ?? "Thinking…")
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(KB.textSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)   // centered in the panel
    }

    // MARK: Result

    private func resultRow(_ result: String) -> some View {
        VStack(spacing: KB.barGap) {
            resultCard(result)     // hugs at the top (not greedy)

            Spacer(minLength: KB.barGap)   // gray space; pushes the buttons to the bottom

            HStack(spacing: KB.barGap) {
                secondary(icon: "xmark", tint: KB.textSecondary) { state.controller?.dismissResult() }
                if let task = state.activeTask {
                    secondary(icon: "arrow.counterclockwise", tint: KB.textPrimary) {
                        state.controller?.runAction(task)
                    }
                }
                Button { state.controller?.insertResult() } label: {
                    Label("Insert result", systemImage: "arrow.down.doc.fill")
                        .font(.system(size: 15, weight: .bold))
                        .frame(maxWidth: .infinity)
                        .frame(height: KB.controlHeight)
                        .foregroundStyle(.white)
                        .background(RoundedRectangle(cornerRadius: KB.radiusControl, style: .continuous).fill(KB.cherry))
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The result preview: hugs its content for short results (so it's not a big
    /// empty box), switches to a fixed-height scroll only for long ones. Never greedy,
    /// so the `Spacer` below can push the buttons to the bottom of the panel.
    @ViewBuilder
    private func resultCard(_ result: String) -> some View {
        let inner = resultText(result).frame(maxWidth: .infinity, alignment: .topLeading)
        Group {
            if result.count > 140 {
                ScrollView { inner }.frame(height: KB.resultMaxHeight)
            } else {
                inner.fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: KB.radiusControl, style: .continuous).fill(KB.keyFill))
    }

    private func resultText(_ result: String) -> some View {
        Text(result)
            .font(.system(size: 15))
            .foregroundStyle(KB.textPrimary)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func secondary(icon: String, tint: Color, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: KB.secondaryWidth, height: KB.controlHeight)
                .background(RoundedRectangle(cornerRadius: KB.radiusControl, style: .continuous).fill(KB.specialFill))
        }
        .buttonStyle(.plain)
    }

    private func card(_ fill: Color) -> some View {
        RoundedRectangle(cornerRadius: KB.radiusControl, style: .continuous)
            .fill(fill)
            .shadow(color: KB.keyShadow, radius: 1, y: 1)
    }
}

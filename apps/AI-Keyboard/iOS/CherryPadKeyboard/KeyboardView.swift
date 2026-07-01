import SwiftUI

/// The CherryPad keyboard UI: AI action bar on top, a standard QWERTY below with
/// shift, delete, plane switching (123/#+=), globe (next keyboard), space, return.
///
/// Height is NOT self-measured here (that deadlocks — content size can't exceed the
/// keyboard height it's measured inside). The controller sizes the keyboard via
/// `UIHostingController.sizeThatFits`, which is independent of the current height.
struct KeyboardView: View {
    @ObservedObject var state: KeyboardState

    /// Idle shows the action bar + QWERTY. Result/processing hide the keys so the AI
    /// panel takes over the SAME fixed area (no overflow past the keyboard).
    private var showKeys: Bool { !state.processing && state.resultText == nil }

    var body: some View {
        VStack(spacing: KB.sectionGap) {
            KeyboardActionBar(state: state)
            if showKeys {
                keysSection
            }
        }
        .padding(.horizontal, KB.sideMargin)
        .padding(.top, KB.sectionGap)
        .padding(.bottom, 4)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .background(KB.background)
    }

    private var keysSection: some View {
        VStack(spacing: KB.keyGapV) {
            ForEach(Array(rows.enumerated()), id: \.offset) { _, row in
                HStack(spacing: KB.keyGapH) {
                    ForEach(row, id: \.self) { key in
                        cap(for: key)
                    }
                }
            }
            bottomRow
        }
    }

    // MARK: Rows

    private var rows: [[String]] {
        switch state.plane {
        case .letters:
            return [
                ["q","w","e","r","t","y","u","i","o","p"],
                ["a","s","d","f","g","h","j","k","l"],
                ["⇧","z","x","c","v","b","n","m","⌫"],
            ]
        case .numbers:
            return [
                ["1","2","3","4","5","6","7","8","9","0"],
                ["-","/",":",";","(",")","$","&","@","\""],
                ["#+=",".",",","?","!","'","⌫"],
            ]
        case .symbols:
            return [
                ["[","]","{","}","#","%","^","*","+","="],
                ["_","\\","|","~","<",">","€","£","¥","•"],
                ["123",".",",","?","!","'","⌫"],
            ]
        }
    }

    private var bottomRow: some View {
        HStack(spacing: 5) {
            KeyCap(label: state.plane == .letters ? "123" : "ABC", fill: KB.specialFill, width: 84, fontSize: 15) {
                state.plane = (state.plane == .letters) ? .numbers : .letters
            }
            KeyCap(systemImage: "globe", fill: KB.specialFill, width: 44) {
                state.controller?.nextKeyboard()
            }
            KeyCap(label: "space", fill: KB.keyFill, fontSize: 15) {
                state.controller?.insert(" ")
            }
            KeyCap(label: "return", fill: KB.specialFill, width: 92, fontSize: 15) {
                state.controller?.newLine()
            }
        }
    }

    // MARK: Key dispatch

    @ViewBuilder
    private func cap(for key: String) -> some View {
        switch key {
        case "⇧":
            KeyCap(systemImage: state.shifted ? "shift.fill" : "shift",
                   fill: state.shifted ? KB.keyFill : KB.specialFill, width: 44) {
                state.shifted.toggle()
            }
        case "⌫":
            KeyCap(systemImage: "delete.left", fill: KB.specialFill, width: 44) {
                state.controller?.deleteBackward()
            }
        case "#+=":
            KeyCap(label: "#+=", fill: KB.specialFill, width: 56, fontSize: 15) {
                state.plane = .symbols
            }
        case "123" where state.plane == .symbols:
            KeyCap(label: "123", fill: KB.specialFill, width: 56, fontSize: 15) {
                state.plane = .numbers
            }
        default:
            let text = state.shifted ? key.uppercased() : key
            KeyCap(label: text, fill: KB.keyFill) {
                state.controller?.insert(text)
            }
        }
    }
}

/// One key. `width` makes it fixed; otherwise it flexes to fill the row.
private struct KeyCap: View {
    var label: String? = nil
    var systemImage: String? = nil
    var fill: Color
    var width: CGFloat? = nil
    var fontSize: CGFloat = 20
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Group {
                if let systemImage {
                    Image(systemName: systemImage).font(.system(size: 17, weight: .regular))
                } else {
                    Text(label ?? "")
                        .font(.system(size: fontSize, weight: .regular))
                        .lineLimit(1).minimumScaleFactor(0.7)
                }
            }
            .foregroundStyle(KB.textPrimary)
            .frame(maxWidth: width == nil ? .infinity : nil)
            .frame(width: width, height: KB.keyHeight)
            .background(
                RoundedRectangle(cornerRadius: KB.radiusKey, style: .continuous)
                    .fill(fill)
                    .shadow(color: KB.keyShadow, radius: 1, y: 1)
            )
        }
        .buttonStyle(.plain)
    }
}

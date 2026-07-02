#if DEBUG
import SwiftUI

/// Simulator-only harness that renders the real `KeyboardView` in each state at the
/// FIXED keyboard height (so it matches the device exactly). Result/processing take
/// over the area (keys hide); idle shows the action bar + QWERTY.
struct KeyboardDesignHarness: View {
    var body: some View {
        ScrollView {
            VStack(spacing: 18) {
                Text("CherryPad keyboard — design harness")
                    .font(.headline).frame(maxWidth: .infinity, alignment: .leading)

                group("Result — short") {
                    keyboard(mock { $0.resultText = "Sure! Let's grab coffee this week."; $0.activeTask = .reply })
                }
                group("Result — long") {
                    keyboard(mock {
                        $0.resultText = "Hello, I appreciate your company and would like to discuss potential opportunities. Could you please review my resume at your earliest convenience? I'm available any afternoon this week."
                        $0.activeTask = .rewrite
                    })
                }
                group("Idle") { keyboard(mock { _ in }) }
                group("Processing") { keyboard(mock { $0.processing = true; $0.statusText = "Thinking…" }) }
            }
            .padding(16)
        }
        .background(Color(.systemGray6))
    }

    private func keyboard(_ state: KeyboardState) -> some View {
        KeyboardView(state: state)
            .frame(height: KB.keyboardHeight)   // the device's fixed keyboard height
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.caption.bold()).foregroundStyle(.secondary)
            content()
        }
    }

    private func mock(_ configure: (KeyboardState) -> Void) -> KeyboardState {
        let s = KeyboardState()
        configure(s)
        return s
    }
}
#endif

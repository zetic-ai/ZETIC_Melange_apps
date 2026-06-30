import UIKit
import SwiftUI
import Combine

/// Observable keyboard state shared with the SwiftUI layout. Holds a weak ref to
/// the controller so key views can drive text operations and AI handoffs.
final class KeyboardState: ObservableObject {
    @Published var plane: KeyPlane = .letters
    @Published var shifted = true            // start with an initial capital
    @Published var resultAvailable = false   // a finished result is waiting to insert
    @Published var banner: String?           // transient hint (e.g. "Select text first")
    @Published var needsFullAccess = false

    weak var controller: KeyboardViewController?
}

enum KeyPlane { case letters, numbers, symbols }

/// CherryPad keyboard: a standard QWERTY plus an AI action bar. It runs NO model
/// (the extension's ~60-120MB budget can't hold one) — actions capture the text,
/// stash it in the shared App Group, and deep-link into the container app, which
/// runs inference and writes the result back for "Insert result".
class KeyboardViewController: UIInputViewController {
    private let state = KeyboardState()
    private var hosting: UIHostingController<KeyboardView>!

    override func viewDidLoad() {
        super.viewDidLoad()
        state.controller = self

        let host = UIHostingController(rootView: KeyboardView(state: state))
        host.view.translatesAutoresizingMaskIntoConstraints = false
        host.view.backgroundColor = .clear
        addChild(host)
        view.addSubview(host.view)
        host.didMove(toParent: self)
        NSLayoutConstraint.activate([
            host.view.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.view.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            host.view.topAnchor.constraint(equalTo: view.topAnchor),
            host.view.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
        hosting = host
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        refreshState()
    }

    override func textDidChange(_ textInput: UITextInput?) {
        super.textDidChange(textInput)
        // Returning from the app: surface a freshly-published result.
        refreshState()
    }

    private func refreshState() {
        state.needsFullAccess = !hasFullAccess
        state.resultAvailable = HandoffStore.readResult() != nil
    }

    // MARK: Text editing

    func insert(_ text: String) {
        textDocumentProxy.insertText(text)
        // Auto-drop shift after a character, like the system keyboard.
        if state.shifted, text != " " { state.shifted = false }
    }

    func deleteBackward() { textDocumentProxy.deleteBackward() }

    func newLine() { textDocumentProxy.insertText("\n") }

    func nextKeyboard() { advanceToNextInputMode() }

    // MARK: AI handoff

    /// Captures the relevant text near the cursor.
    private func capturedText() -> String {
        if let selected = textDocumentProxy.selectedText, !selected.isEmpty {
            return selected
        }
        let before = textDocumentProxy.documentContextBeforeInput ?? ""
        let after = textDocumentProxy.documentContextAfterInput ?? ""
        return (before + after).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Handles an AI action: stash a request and launch the container app.
    func runAction(_ task: KeyboardTask) {
        guard hasFullAccess else {
            state.banner = "Enable Full Access: Settings ▸ Keyboards ▸ CherryPad."
            return
        }
        let text = capturedText()
        guard !text.isEmpty else {
            state.banner = "Type or select text first, then tap again."
            return
        }
        let request = HandoffRequest(
            task: task,
            text: text,
            tone: task == .rewrite ? .professional : nil,
            stance: task == .reply ? .agreeable : nil
        )
        HandoffStore.clearResult()
        HandoffStore.writeRequest(request)
        // Fallback: copy the input so manual app-open still has the text if the
        // deep link is blocked by the OS.
        UIPasteboard.general.string = text
        state.banner = "Opening CherryPad…"
        if let url = DeepLink.process(id: request.id) {
            openHostApp(url)
        }
    }

    /// Inserts the result the app published, replacing the current selection.
    func insertResult() {
        guard let result = HandoffStore.readResult() else { return }
        if let selected = textDocumentProxy.selectedText, !selected.isEmpty {
            textDocumentProxy.deleteBackward()
        }
        textDocumentProxy.insertText(result.text)
        HandoffStore.clearResult()
        state.resultAvailable = false
    }

    // MARK: openURL from a keyboard extension (responder-chain trick)

    @objc func openURL(_ url: URL) {}

    private func openHostApp(_ url: URL) {
        var responder: UIResponder? = self
        while let current = responder {
            if let app = current as? UIApplication, app.responds(to: #selector(openURL(_:))) {
                app.perform(#selector(openURL(_:)), with: url)
                return
            }
            responder = current.next
        }
        // Fallback: nothing in the chain could open URLs — the user can still open
        // CherryPad manually; the request + text are already saved.
        state.banner = "Open CherryPad to finish (text is copied)."
    }
}

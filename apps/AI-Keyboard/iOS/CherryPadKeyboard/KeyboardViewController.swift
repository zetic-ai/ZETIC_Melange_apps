import UIKit
import SwiftUI
import Combine

/// CherryPad keyboard. Runs the model IN-PROCESS (KeyboardLLM) so AI actions work
/// entirely on the keyboard — no app round-trip. Tap an action → it generates →
/// "Insert result" drops it in.
class KeyboardViewController: UIInputViewController, KeyboardActions {
    private let state = KeyboardState()
    private var hosting: UIHostingController<KeyboardView>!
    private var hadSelection = false

    private var heightConstraint: NSLayoutConstraint!

    override func viewDidLoad() {
        super.viewDidLoad()
        state.controller = self
        view.backgroundColor = UIColor.systemGray5   // matches KB.background; avoids white gaps
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
        // FIXED height = the idle content height (action bar + QWERTY). The result /
        // processing states DON'T grow the keyboard (that never worked reliably —
        // extensions resist resizing); instead the AI panel takes over this same fixed
        // area (the QWERTY hides), so the output always fits — no overflow/clipping.
        heightConstraint = view.heightAnchor.constraint(equalToConstant: KB.keyboardHeight)
        heightConstraint.priority = UILayoutPriority(999)
        heightConstraint.isActive = true
        hosting = host
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        state.needsFullAccess = !hasFullAccess
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Free the model under pressure so the next action reloads instead of the
        // extension being jetsam-killed mid-use.
        KeyboardLLM.shared.unload()
    }

    // MARK: Text editing

    func insert(_ text: String) {
        textDocumentProxy.insertText(text)
        if state.shifted, text != " " { state.shifted = false }
    }
    func deleteBackward() { textDocumentProxy.deleteBackward() }
    func newLine() { textDocumentProxy.insertText("\n") }
    func nextKeyboard() { advanceToNextInputMode() }

    // MARK: AI (runs in-keyboard)

    private func capturedText() -> String {
        if let selected = textDocumentProxy.selectedText, !selected.isEmpty { return selected }
        let before = textDocumentProxy.documentContextBeforeInput ?? ""
        let after = textDocumentProxy.documentContextAfterInput ?? ""
        return (before + after).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func runAction(_ task: KeyboardTask) {
        guard !state.processing else { return }
        guard hasFullAccess else {
            state.banner = "Enable Full Access: Settings ▸ Keyboards ▸ CherryPad."
            return
        }
        let selected = textDocumentProxy.selectedText
        hadSelection = (selected != nil && !(selected ?? "").isEmpty)
        let text = capturedText()
        guard !text.isEmpty else {
            state.banner = "Type or select text first, then tap again."
            return
        }

        state.banner = nil
        state.resultText = nil
        state.activeTask = task
        state.processing = true
        state.statusText = "Preparing…"

        let targetLang = AppGroup.defaults.string(forKey: "cherrypad.targetLang") ?? "Korean"
        KeyboardLLM.shared.generate(
            task: task,
            text: text,
            tone: task == .rewrite ? .professional : nil,
            stance: task == .reply ? .agreeable : nil,
            targetLanguage: task == .translate ? targetLang : nil,
            onStatus: { status in
                DispatchQueue.main.async {
                    switch status {
                    case .downloading(let p):
                        self.state.statusText = (p > 0 && p < 1) ? "Downloading model… \(Int(p * 100))%" : "Preparing…"
                    case .thinking:
                        self.state.statusText = "Thinking…"
                    }
                }
            },
            completion: { result in
                DispatchQueue.main.async {
                    self.state.processing = false
                    self.state.statusText = nil
                    switch result {
                    case .success(let text):
                        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            self.state.banner = "No result — try again."
                        } else {
                            self.state.resultText = text
                        }
                    case .failure(let error):
                        self.state.banner = "Failed: \(error.localizedDescription)"
                    }
                }
            }
        )
    }

    /// Inserts the prepared result, replacing the original selection if there was one.
    func insertResult() {
        guard let text = state.resultText else { return }
        if hadSelection { textDocumentProxy.deleteBackward() } // removes the selection
        textDocumentProxy.insertText(text)
        state.resultText = nil
        state.activeTask = nil
    }

    func dismissResult() {
        state.resultText = nil
        state.activeTask = nil
        state.banner = nil
    }
}

import SwiftUI

/// Actions the SwiftUI keyboard views trigger. Abstracting these behind a protocol
/// (instead of referencing `KeyboardViewController` directly) keeps the views free
/// of any UIInputViewController / SDK dependency, so they compile in the Simulator
/// preview app for visual iteration.
protocol KeyboardActions: AnyObject {
    func insert(_ text: String)
    func deleteBackward()
    func newLine()
    func nextKeyboard()
    func runAction(_ task: KeyboardTask)
    func insertResult()
    func dismissResult()
}

enum KeyPlane { case letters, numbers, symbols }

/// Observable state shared between the controller and the SwiftUI layout.
final class KeyboardState: ObservableObject {
    @Published var plane: KeyPlane = .letters
    @Published var shifted = true
    @Published var needsFullAccess = false
    @Published var processing = false        // an AI action is running
    @Published var statusText: String?       // "Downloading model… 40%", "Thinking…"
    @Published var resultText: String?       // ready-to-insert AI result
    @Published var activeTask: KeyboardTask?  // which action produced the result
    @Published var banner: String?           // transient hint / error

    weak var controller: KeyboardActions?
}

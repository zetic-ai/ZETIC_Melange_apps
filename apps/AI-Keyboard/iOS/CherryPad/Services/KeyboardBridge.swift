import UIKit

/// App-side helpers for the keyboard handoff: publishing a finished result back to
/// the shared App Group (so the keyboard's "Insert result" can pick it up) and
/// copying to the system pasteboard (so the user can paste in the host app).
enum KeyboardBridge {
    /// Writes the result to the App Group for a pending keyboard request.
    static func publishResult(requestID: UUID, text: String) {
        HandoffStore.writeResult(HandoffResult(requestID: requestID, text: text))
    }

    /// Copies text to the system pasteboard.
    static func copyToPasteboard(_ text: String) {
        UIPasteboard.general.string = text
    }
}

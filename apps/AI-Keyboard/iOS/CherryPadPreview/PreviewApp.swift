import SwiftUI

/// `@main` for the Simulator-only `CherryPadPreview` target (the device app uses
/// `CherryPadApp`, which this target excludes). Launch with `-kbDesign` to render the
/// keyboard design harness for visual iteration; otherwise the normal app.
@main
struct PreviewApp: App {
    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.arguments.contains("-kbDesign") {
                KeyboardDesignHarness()
            } else {
                RootView()
            }
        }
    }
}

import SwiftUI

@main
struct ImageTo3DApp: App {
    init() {
        if CommandLine.arguments.contains("--selftest") {
            SelfTest.shared.start()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .preferredColorScheme(.dark)
        }
    }
}

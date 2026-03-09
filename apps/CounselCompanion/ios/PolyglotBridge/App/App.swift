import SwiftUI

@main
struct PolyglotBridgeApp: App {
    @StateObject private var appViewModel = AppViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(appViewModel)
                .preferredColorScheme(appViewModel.settings.theme.colorScheme)
        }
    }
}

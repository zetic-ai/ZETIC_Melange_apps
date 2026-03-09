import SwiftUI

@main
struct CounselCompanionApp: App {
    @StateObject private var viewModel = ChatViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(viewModel)
                .preferredColorScheme(viewModel.themeOverride)
        }
    }
}

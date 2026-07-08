import SwiftUI

@main
struct SkinImageClassificationApp: App {
    @StateObject private var viewModel = DiagnosisViewModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(viewModel)
                .preferredColorScheme(.dark)
                .tint(Theme.accent)
        }
    }
}

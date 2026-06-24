import SwiftUI
import UIKit

@main
struct VoiceVitalsApp: App {
    @StateObject private var models = AppModels()

    init() { Self.configureAppearance() }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(models)
                .task { models.preloadAll() }   // download every model at launch
        }
    }

    /// Match the warm cream theme on the tab bar.
    private static func configureAppearance() {
        let cream = UIColor(red: 0.953, green: 0.945, blue: 0.918, alpha: 1)
        let tab = UITabBarAppearance()
        tab.configureWithOpaqueBackground()
        tab.backgroundColor = cream
        UITabBar.appearance().standardAppearance = tab
        UITabBar.appearance().scrollEdgeAppearance = tab
    }
}

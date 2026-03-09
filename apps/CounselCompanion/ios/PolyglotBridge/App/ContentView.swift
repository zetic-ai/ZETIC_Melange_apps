import SwiftUI

enum MainTab { case translate, history, favorites, settings, diagnostics }

struct ContentView: View {
    @EnvironmentObject private var app: AppViewModel
    @State private var tab: MainTab = .translate

    var body: some View {
        TabView(selection: $tab) {
            TranslateScreen()
                .tabItem { Label("Translate", systemImage: "bubble.left.and.bubble.right.fill") }
                .tag(MainTab.translate)

            HistoryScreen(onSelect: { session in
                app.resumeSession(session.id)
                tab = .translate
            })
            .tabItem { Label("History", systemImage: "clock.fill") }
            .tag(MainTab.history)

            FavoritesScreen(onUse: { phrase in
                tab = .translate
                app.updateSettings { _ in }
                NotificationCenter.default.post(name: .useFavorite, object: phrase)
            })
            .tabItem { Label("Favorites", systemImage: "star.fill") }
            .tag(MainTab.favorites)

            SettingsScreen()
                .tabItem { Label("Settings", systemImage: "gearshape.fill") }
                .tag(MainTab.settings)

            DiagnosticsScreen()
                .tabItem { Label("Diagnostics", systemImage: "waveform.path.ecg") }
                .tag(MainTab.diagnostics)
        }
        .task {
            // auto prepare on first launch if not ready
            if !app.modelReady {
                await app.prepareModel()
            }
        }
    }
}

extension Notification.Name {
    static let useFavorite = Notification.Name("useFavorite")
}

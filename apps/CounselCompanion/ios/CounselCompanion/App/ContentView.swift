import SwiftUI

enum MainTab: Hashable { case chat, sessions, settings, diagnostics }

struct ContentView: View {
    @EnvironmentObject var viewModel: ChatViewModel
    @State private var tab: MainTab = .chat

    var body: some View {
        TabView(selection: $tab) {
            ChatScreen()
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                .tag(MainTab.chat)

            SessionsScreen()
                .tabItem { Label("Sessions", systemImage: "clock") }
                .tag(MainTab.sessions)

            SettingsScreen()
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(MainTab.settings)

            DiagnosticsScreen()
                .tabItem { Label("Diagnostics", systemImage: "waveform.path.ecg") }
                .tag(MainTab.diagnostics)
        }
        .tint(Color.warmSage)
        .task { await viewModel.bootstrap() }
    }
}

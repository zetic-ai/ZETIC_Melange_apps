//
//  ContentView.swift
//  PromptGuard
//

import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            LiveView()
                .tabItem { Label("Classify", systemImage: "shield.checkered") }
                .tag(0)
            HistoryView()
                .tabItem { Label("History", systemImage: "clock.arrow.circlepath") }
                .tag(1)
            SettingsView()
                .tabItem { Label("Settings", systemImage: "gearshape") }
                .tag(2)
            DiagnosticsView()
                .tabItem { Label("Diagnostics", systemImage: "stethoscope") }
                .tag(3)
        }
        .tint(AppTheme.accent)
    }
}

#Preview {
    ContentView()
}

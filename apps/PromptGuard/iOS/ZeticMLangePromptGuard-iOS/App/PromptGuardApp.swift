//
//  PromptGuardApp.swift
//  PromptGuard
//
//  Analyzes and classifies prompt injection and jailbreak attacks.
//

import SwiftUI

@main
struct PromptGuardApp: App {
    @AppStorage("useDarkTheme") private var useDarkTheme = false

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
                .ignoresSafeArea(.all)
                .tint(AppTheme.accent)
                .preferredColorScheme(useDarkTheme ? .dark : .light)
        }
    }
}


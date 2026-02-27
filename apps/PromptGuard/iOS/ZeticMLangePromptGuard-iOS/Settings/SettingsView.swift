//
//  SettingsView.swift
//  PromptGuard
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("useDarkTheme") private var useDarkTheme = true
    @AppStorage("dataRetentionDays") private var dataRetentionDays = 30
    @State private var showClearHistoryAlert = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Toggle("Dark theme", isOn: $useDarkTheme)
                } header: {
                    Text("Appearance")
                }
                Section {
                    Picker("Data retention", selection: $dataRetentionDays) {
                        Text("7 days").tag(7)
                        Text("30 days").tag(30)
                        Text("90 days").tag(90)
                        Text("Keep all").tag(3650)
                    }
                    Button("Clear history", role: .destructive) {
                        showClearHistoryAlert = true
                    }
                } header: {
                    Text("Privacy & data")
                } footer: {
                    Text("Classification history is stored only on this device. Clearing history removes all saved entries. Adjust ModelInputSpec in Diagnostics.")
                }
                Section {
                    Text("PromptGuard analyzes prompts for harmful content (S1–S11). Inference runs on-device via Zetic Melange. No prompt text or keys are sent to external servers. Your personal key is used only to download the model and run it locally.")
                        .font(.caption)
                } header: {
                    Text("Privacy explanation")
                }
            }
            .navigationTitle("Settings")
            .alert("Clear history?", isPresented: $showClearHistoryAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Clear", role: .destructive) {
                    HistoryStore.shared.clear()
                }
            } message: {
                Text("All classification history will be removed.")
            }
        }
    }
}

#Preview {
    SettingsView()
}

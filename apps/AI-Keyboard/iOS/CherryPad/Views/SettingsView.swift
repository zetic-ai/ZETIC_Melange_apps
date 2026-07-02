import SwiftUI

/// Settings: enable-keyboard guide + model info.
struct SettingsView: View {
    @ObservedObject var llm: LLMService
    var onShowOnboarding: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Button("How to enable the keyboard") { onShowOnboarding() }
                        .foregroundStyle(Theme.cherry)
                }

                Section {
                    LabeledContent("Model", value: "LFM2.5 350M")
                    LabeledContent("Runs", value: "100% on-device")
                } header: {
                    Text("About")
                } footer: {
                    Text("A small on-device model powers Rewrite, Reply, Translate, and Grammar — right on the keyboard, no network needed after the first download.")
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.foregroundStyle(Theme.cherry)
                }
            }
        }
    }
}

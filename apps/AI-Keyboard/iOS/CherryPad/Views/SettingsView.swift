import SwiftUI

/// Settings: speed/quality tier and a link back to the enable-keyboard guide.
struct SettingsView: View {
    @ObservedObject var llm: LLMService
    var onShowOnboarding: () -> Void
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(ZeticConfig.Quality.allCases) { quality in
                        Button {
                            Task { await llm.setQuality(quality) }
                        } label: {
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(quality.label)
                                        .font(.system(size: 16, weight: .medium))
                                        .foregroundStyle(Theme.textPrimary)
                                    Text(quality.detail)
                                        .font(.system(size: 12))
                                        .foregroundStyle(Theme.textSecondary)
                                }
                                Spacer()
                                if quality == llm.quality {
                                    Image(systemName: "checkmark")
                                        .font(.system(size: 14, weight: .bold))
                                        .foregroundStyle(Theme.cherry)
                                }
                            }
                        }
                    }
                } header: {
                    Text("Model speed")
                } footer: {
                    Text("Fast runs the smallest model for near-instant results. Higher quality is more nuanced but slower and uses more memory. Switching re-downloads the other model once.")
                }

                Section {
                    Button("How to enable the keyboard") { onShowOnboarding() }
                        .foregroundStyle(Theme.cherry)
                }

                Section {
                    LabeledContent("Model", value: llm.quality.modelName)
                    LabeledContent("Runs", value: "100% on-device")
                } header: {
                    Text("About")
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

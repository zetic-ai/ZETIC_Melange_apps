import SwiftUI

struct SettingsScreen: View {
    @EnvironmentObject private var app: AppViewModel

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Appearance")) {
                    Picker("Theme", selection: Binding(get: { app.settings.theme }, set: { newValue in app.updateSettings { $0.theme = newValue } })) {
                        ForEach(SettingsData.Theme.allCases, id: \.self) { theme in
                            Text(theme.label).tag(theme)
                        }
                    }
                }

                Section(header: Text("Defaults")) {
                    Picker("Default source", selection: Binding(get: { app.settings.defaultSource }, set: { newValue in app.updateSettings { $0.defaultSource = newValue } })) {
                        ForEach(app.languages) { lang in
                            Text(lang.name).tag(lang.code)
                        }
                    }
                    Picker("Default target", selection: Binding(get: { app.settings.defaultTarget }, set: { newValue in app.updateSettings { $0.defaultTarget = newValue } })) {
                        ForEach(app.languages) { lang in
                            Text(lang.name).tag(lang.code)
                        }
                    }
                }

                Section(header: Text("System Prompt")) {
                    TextEditor(text: Binding(get: { app.settings.systemPrompt }, set: { newValue in app.updateSettings { $0.systemPrompt = newValue } }))
                        .frame(minHeight: 80)
                }

                Section(header: Text("Data")) {
                    Button("Clear history") { app.clearHistory() }
                    Button("Clear favorites") { app.clearFavorites() }
                }

                Section(header: Text("Model")) {
                    HStack {
                        Text("Ready state")
                        Spacer()
                        Text(app.modelReady ? "Ready" : "Not ready")
                            .foregroundColor(app.modelReady ? .green : .orange)
                    }
                    Button("Prepare / Download again") {
                        Task { await app.prepareModel() }
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

import SwiftUI

struct DiagnosticsScreen: View {
    @EnvironmentObject private var app: AppViewModel

    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Model")) {
                    HStack { Text("Model ID"); Spacer(); Text(TranslationEngine.modelName).multilineTextAlignment(.trailing) }
                    HStack { Text("Personal key"); Spacer(); Text(TranslationEngine.maskedKey()) }
                    HStack { Text("Ready"); Spacer(); Text(app.modelReady ? "Yes" : "No").foregroundColor(app.modelReady ? .green : .orange) }
                }

                Section(header: Text("Last run")) {
                    HStack { Text("Duration"); Spacer(); Text("\(app.diagnostics.lastDurationMs) ms") }
                    HStack { Text("Tokens"); Spacer(); Text("\(app.diagnostics.lastTokenCount)") }
                    HStack { Text("Stop reason"); Spacer(); Text(app.diagnostics.lastStopReason) }
                    if let date = app.diagnostics.lastRun {
                        HStack { Text("At"); Spacer(); Text(date.formatted(date: .abbreviated, time: .standard)) }
                    }
                }

                Section(header: Text("Raw log")) {
                    if app.diagnostics.lastRawLog.isEmpty {
                        Text("No log yet")
                            .foregroundColor(.secondary)
                    } else {
                        ScrollView {
                            Text(app.diagnostics.lastRawLog)
                                .font(AppFont.mono())
                        }
                        .frame(maxHeight: 240)
                    }
                }
            }
            .navigationTitle("Diagnostics")
        }
    }
}

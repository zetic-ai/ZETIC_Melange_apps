//
//  DiagnosticsView.swift
//  PromptGuard
//

import SwiftUI

struct DiagnosticsView: View {
    @StateObject private var model = PromptGuardModel()
    @State private var spec: ModelInputSpec = ModelInputSpecStore.shared.spec
    @State private var copied = false

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Text classification (prompt guard)")
                            .font(.subheadline.weight(.medium))
                        Text("Model: jathin-zetic/llama_prompt_guard_2. Input: tokenized prompt (UTF-8 byte encoding). Output: scores per harm category.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                } header: {
                    Text("Inferred modality")
                }
                Section {
                    Stepper("Max tokens: \(spec.maxTokens)", value: $spec.maxTokens, in: 64...2048, step: 64)
                        .onChange(of: spec.maxTokens) { _ in
                            ModelInputSpecStore.shared.spec = spec
                        }
                    TextField("Prompt template", text: $spec.promptTemplate, axis: .vertical)
                        .lineLimit(2...4)
                        .onChange(of: spec.promptTemplate) { _ in
                            ModelInputSpecStore.shared.spec = spec
                        }
                } header: {
                    Text("ModelInputSpec")
                } footer: {
                    Text("Use {user_input} and {agent_output} in the template.")
                }
                Section {
                    if let ms = model.lastLatencyMs {
                        Label(String(format: "Last latency: %.0f ms", ms), systemImage: "clock")
                    }
                    if let err = model.lastError {
                        Label(err, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(AppTheme.danger)
                    }
                    if model.lastError == nil && model.lastLatencyMs == nil {
                        Text("Run a classification on the Classify tab to see telemetry.")
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Telemetry")
                }
                Section {
                    if let raw = model.lastRawOutput, !raw.isEmpty {
                        Text(raw)
                            .font(.caption.monospaced())
                            .textSelection(.enabled)
                        Button {
                            UIPasteboard.general.string = raw
                            copied = true
                            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                        } label: {
                            Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc")
                        }
                    } else {
                        Text("No output yet. Run a classification to see raw output.")
                            .foregroundStyle(.secondary)
                    }
                } header: {
                    Text("Raw output (first values)")
                }
            }
            .navigationTitle("Diagnostics")
            .onAppear {
                spec = ModelInputSpecStore.shared.spec
                model.load()
            }
        }
    }

}

#Preview {
    DiagnosticsView()
}

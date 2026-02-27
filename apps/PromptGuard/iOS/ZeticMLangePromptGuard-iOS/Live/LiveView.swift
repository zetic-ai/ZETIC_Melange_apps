//
//  LiveView.swift
//  PromptGuard
//

import SwiftUI

struct LiveView: View {
    @StateObject private var model = PromptGuardModel()
    @State private var userInput = ""
    @State private var agentOutput = ""
    @State private var result: ClassificationResult?
    @State private var isRunning = false
    @AppStorage("useDarkTheme") private var useDarkTheme = true

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    inputSection
                    if isRunning {
                        ProgressView("Classifying…")
                            .padding()
                    }
                    if let r = result {
                        resultSection(r)
                    }
                    if let err = model.lastError {
                        errorBanner(err)
                    }
                    telemetrySection
                }
                .padding()
            }
            .navigationTitle("Classify")
            .background(Color(.systemGroupedBackground))
            .onAppear {
                model.load()
            }
        }
    }

    private var inputSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("User prompt")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
            TextField("Enter prompt to classify…", text: $userInput, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(3...8)
            Text("Agent output (optional)")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
            TextField("Agent response for context", text: $agentOutput, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(2...4)
            Button {
                runClassification()
            } label: {
                Label("Classify", systemImage: "shield.checkered")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
            }
            .buttonStyle(.borderedProminent)
            .tint(AppTheme.accent)
            .disabled(userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isRunning)
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private func resultSection(_ r: ClassificationResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Harm categories")
                .font(.headline)
            let categories = HarmCategory.allCases
            let pairs = Array(zip(categories, r.categoryScores))
            LazyVStack(spacing: 8) {
                ForEach(Array(pairs.enumerated()), id: \.offset) { _, pair in
                    let (cat, score) = pair
                    HStack {
                        Text(cat.rawValue)
                            .font(.caption.monospaced())
                            .frame(width: 28, alignment: .leading)
                        Text(cat.displayName)
                            .font(.subheadline)
                        Spacer()
                        Text(String(format: "%.2f", score))
                            .font(.subheadline.monospacedDigit())
                            .foregroundStyle(score > 0.5 ? AppTheme.danger : .secondary)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        score > 0.3 ? AppTheme.danger.opacity(0.15) : Color.clear,
                        in: RoundedRectangle(cornerRadius: 8)
                    )
                }
            }
        }
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
    }

    private func errorBanner(_ message: String) -> some View {
        HStack {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(AppTheme.danger)
            Text(message)
                .font(.subheadline)
                .foregroundStyle(.primary)
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(AppTheme.danger.opacity(0.2), in: RoundedRectangle(cornerRadius: 12))
    }

    private var telemetrySection: some View {
        HStack {
            if let ms = model.lastLatencyMs {
                Label(String(format: "%.0f ms", ms), systemImage: "clock")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            if model.lastError != nil {
                Label("Error", systemImage: "xmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(AppTheme.danger)
            }
        }
        .padding(.horizontal)
    }

    private func runClassification() {
        let input = userInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return }
        isRunning = true
        result = nil
        Task {
            let r = await model.classify(userInput: input, agentOutput: agentOutput.trimmingCharacters(in: .whitespacesAndNewlines))
            await MainActor.run {
                isRunning = false
                result = r
                if let r = r {
                    let topIdx = r.categoryScores.enumerated().max(by: { $0.element < $1.element }).map(\.offset) ?? 0
                    let cat = HarmCategory.allCases[topIdx]
                    HistoryStore.shared.add(entry: HistoryEntry(
                        id: UUID(),
                        date: Date(),
                        userInputPreview: String(input.prefix(80)),
                        topCategory: cat.rawValue,
                        topScore: r.categoryScores[topIdx],
                        latencyMs: model.lastLatencyMs,
                        allScores: r.categoryScores
                    ))
                    let impact = UIImpactFeedbackGenerator(style: .medium)
                    impact.impactOccurred()
                }
            }
        }
    }
}

#Preview {
    LiveView()
}

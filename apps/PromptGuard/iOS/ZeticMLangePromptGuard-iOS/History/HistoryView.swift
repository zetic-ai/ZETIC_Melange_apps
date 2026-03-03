//
//  HistoryView.swift
//  PromptGuard
//

import SwiftUI
import Charts

struct HistoryView: View {
    @Binding var showSettings: Bool
    @Binding var showDiagnostics: Bool
    @ObservedObject private var store = HistoryStore.shared

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                // Title and settings menu on one row (shifted down from top)
                HStack(alignment: .center) {
                    Text("History")
                        .font(.system(size: 40, weight: .bold))
                        .foregroundStyle(.white)
                    Spacer(minLength: 12)
                    Menu {
                        Button { showSettings = true } label: { Label("Settings", systemImage: "gearshape") }
                        Button { showDiagnostics = true } label: { Label("Diagnostics", systemImage: "stethoscope") }
                    } label: {
                        Image(systemName: "ellipsis.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.white)
                            .padding(12)
                            .background(.ultraThinMaterial, in: Circle())
                            .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 68)

                // Main content
                VStack(alignment: .leading, spacing: 24) {
                    if !store.entries.isEmpty {
                        chartSection
                    }
                    listSection
                }
                .padding(16)
                .padding(.bottom, 24)
                .padding(.bottom, 88) // room for floating tab bar
            }
        }
        .scrollContentBackground(.hidden)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Classifications by category (last 100)")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(AppTheme.textSecondary)
            let data = store.categoryCounts(limit: 100)
            Chart {
                ForEach(Array(data.enumerated()), id: \.offset) { _, item in
                    BarMark(
                        x: .value("Count", item.count),
                        y: .value("Category", item.category)
                    )
                    .foregroundStyle(AppTheme.accent.gradient)
                }
            }
            .frame(height: 220)
        }
        .padding(20)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    private var listSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Recent")
                    .font(.headline)
                Spacer()
                if !store.entries.isEmpty {
                    Button("Clear", role: .destructive) {
                        store.clear()
                    }
                    .font(.subheadline)
                }
            }
            if store.entries.isEmpty {
                Text("No classifications yet. Run a classification on the Classify tab.")
                    .font(.subheadline)
                    .foregroundStyle(AppTheme.textSecondary)
                    .padding()
                    .frame(maxWidth: .infinity)
            } else {
                LazyVStack(spacing: 0) {
                    ForEach(store.entries.prefix(50)) { e in
                        HistoryRow(entry: e)
                    }
                }
            }
        }
        .padding(20)
        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}

struct HistoryRow: View {
    let entry: HistoryEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(entry.userInputPreview)
                .font(.subheadline)
                .lineLimit(2)
            HStack {
                Text(displayCategory(entry.topCategory))
                    .font(.caption.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(AppTheme.accent.opacity(0.3), in: Capsule())
                Text(String(format: "%.2f", entry.topScore))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(AppTheme.textSecondary)
                if let ms = entry.latencyMs {
                    Text("\(Int(ms)) ms")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                Text(entry.date, style: .relative)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 4)
        Divider()
    }

    /// Show only Benign / Malicious; legacy S1–S11 are shown as Malicious.
    private func displayCategory(_ topCategory: String) -> String {
        topCategory == "Benign" ? "Benign" : "Malicious"
    }
}

#Preview {
    HistoryView(showSettings: .constant(false), showDiagnostics: .constant(false))
}

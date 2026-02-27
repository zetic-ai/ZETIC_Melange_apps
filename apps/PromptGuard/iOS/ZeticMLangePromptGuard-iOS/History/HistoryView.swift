//
//  HistoryView.swift
//  PromptGuard
//

import SwiftUI
import Charts

struct HistoryView: View {
    @ObservedObject private var store = HistoryStore.shared

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    if !store.entries.isEmpty {
                        chartSection
                    }
                    listSection
                }
                .padding()
            }
            .navigationTitle("History")
            .background(Color(.systemGroupedBackground))
        }
    }

    private var chartSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Classifications by category (last 100)")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.secondary)
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
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
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
                    .foregroundStyle(.secondary)
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
        .padding()
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 12))
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
                Text(entry.topCategory)
                    .font(.caption.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(AppTheme.accent.opacity(0.3), in: Capsule())
                Text(String(format: "%.2f", entry.topScore))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
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
}

#Preview {
    HistoryView()
}

//
//  HistoryStore.swift
//  PromptGuard
//

import Foundation

struct HistoryEntry: Identifiable, Codable {
    var id: UUID
    var date: Date
    var userInputPreview: String
    var topCategory: String
    var topScore: Float
    var latencyMs: Double?
    var allScores: [Float]
}

final class HistoryStore: ObservableObject {
    static let shared = HistoryStore()
    private let key = "promptguard_history"
    private let maxEntries = 500
    private let defaults = UserDefaults.standard

    @Published var entries: [HistoryEntry] = []

    init() {
        load()
    }

    func add(entry: HistoryEntry) {
        entries.insert(entry, at: 0)
        if entries.count > maxEntries {
            entries = Array(entries.prefix(maxEntries))
        }
        save()
    }

    func clear() {
        entries = []
        save()
    }

    private func load() {
        guard let data = defaults.data(forKey: key),
              let decoded = try? JSONDecoder().decode([HistoryEntry].self, from: data) else { return }
        entries = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(entries) else { return }
        defaults.set(data, forKey: key)
    }

    /// Chart data: count per category over last N entries.
    func categoryCounts(limit: Int = 100) -> [(category: String, count: Int)] {
        let slice = Array(entries.prefix(limit))
        var counts: [String: Int] = [:]
        for e in slice {
            counts[e.topCategory, default: 0] += 1
        }
        return HarmCategory.allCases.map { ($0.rawValue + " " + $0.displayName, counts[$0.rawValue] ?? 0) }
    }
}

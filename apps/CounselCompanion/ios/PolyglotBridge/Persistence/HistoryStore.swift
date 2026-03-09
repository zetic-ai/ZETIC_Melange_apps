import Foundation

final class HistoryStore {
    private let url: URL
    private let queue = DispatchQueue(label: "history.store.queue", qos: .utility)
    private let limit = 50

    init() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        url = dir.appendingPathComponent("polyglot_history.json")
    }

    func load() -> [Session] {
        queue.sync {
            guard let data = try? Data(contentsOf: url) else { return [] }
            return (try? JSONDecoder().decode([Session].self, from: data)) ?? []
        }
    }

    func save(_ sessions: [Session]) {
        queue.async {
            let trimmed = Array(sessions.prefix(self.limit))
            guard let data = try? JSONEncoder().encode(trimmed) else { return }
            try? data.write(to: self.url, options: .atomic)
        }
    }

    func clear() {
        queue.async { try? FileManager.default.removeItem(at: self.url) }
    }
}

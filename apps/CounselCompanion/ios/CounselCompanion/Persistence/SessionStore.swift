import Foundation

final class SessionStore {
    private let url: URL
    private let queue = DispatchQueue(label: "session.store.queue", qos: .utility)

    init() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        url = dir.appendingPathComponent("sessions.json")
    }

    func load() -> [Session] {
        queue.sync {
            guard let data = try? Data(contentsOf: url) else { return [] }
            return (try? JSONDecoder().decode([Session].self, from: data)) ?? []
        }
    }

    func save(_ sessions: [Session]) {
        queue.async {
            guard let data = try? JSONEncoder().encode(sessions) else { return }
            try? data.write(to: self.url, options: .atomic)
        }
    }

    func clear() {
        queue.async { try? FileManager.default.removeItem(at: self.url) }
    }
}

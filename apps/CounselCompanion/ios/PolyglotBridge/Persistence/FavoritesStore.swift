import Foundation

final class FavoritesStore {
    private let url: URL
    private let queue = DispatchQueue(label: "favorites.store.queue", qos: .utility)

    init() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        url = dir.appendingPathComponent("polyglot_favorites.json")
    }

    func load() -> [FavoritePhrase] {
        queue.sync {
            guard let data = try? Data(contentsOf: url) else { return [] }
            return (try? JSONDecoder().decode([FavoritePhrase].self, from: data)) ?? []
        }
    }

    func save(_ favorites: [FavoritePhrase]) {
        queue.async {
            guard let data = try? JSONEncoder().encode(favorites) else { return }
            try? data.write(to: self.url, options: .atomic)
        }
    }

    func clear() {
        queue.async { try? FileManager.default.removeItem(at: self.url) }
    }
}

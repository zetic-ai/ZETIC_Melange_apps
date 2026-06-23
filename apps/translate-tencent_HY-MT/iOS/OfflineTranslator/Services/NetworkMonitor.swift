import Foundation
import Network

/// Live network reachability. Drives the header status pill so it reflects the *real*
/// connection — the whole point of the demo is that translation keeps working when this
/// flips to offline (e.g. Airplane Mode).
@MainActor
final class NetworkMonitor: ObservableObject {
    @Published private(set) var isOnline = true

    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "ai.zetic.translate.network", qos: .utility)

    init() {
        monitor.pathUpdateHandler = { [weak self] path in
            let online = path.status == .satisfied
            Task { @MainActor in self?.isOnline = online }
        }
        monitor.start(queue: queue)
    }

    deinit {
        monitor.cancel()
    }
}
